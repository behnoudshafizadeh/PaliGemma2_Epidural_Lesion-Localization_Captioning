"""
paligemma_resume.py -- adds real resume support on top of maestro's PaliGemma-2
trainer, which doesn't have any.

maestro's own `SaveCheckpoint` callback (maestro.trainer.common.callbacks) only
saves DEPLOYABLE model weights each epoch (processor.save_pretrained() +
model.save_pretrained()) -- no optimizer state, no LR-scheduler state, no epoch/
global-step counter. Its own `train()` also calls `trainer.fit(pl_module)` with no
`ckpt_path`, so there is no code path anywhere in maestro to resume a stopped run;
restarting always means a brand-new, randomly-initialized LoRA adapter
(maestro.trainer.models.paligemma_2.checkpoints.load_model calls
`get_peft_model(model, lora_config)` unconditionally, ignoring any prior training).

Fix: use Lightning's own native checkpointing, which IS designed for this --
`trainer.save_checkpoint(path)` captures the full training state (model weights,
optimizer moments, epoch, global step, RNG state), and `trainer.fit(pl_module,
ckpt_path=path)` restores all of it and continues seamlessly. See
train_paligemma_python.py, which wires this in alongside maestro's own
SaveCheckpoint (kept for the small, deployable adapter-only artifact used for
inference).

Caveats, confirmed against this project's code/config:
- Resume is only at EPOCH BOUNDARIES (this callback fires on_train_epoch_end, same
  as maestro's own). A run killed mid-epoch loses that partial epoch, same as
  before -- completed epochs are never lost.
- The saved file is large: Lightning checkpoints the LightningModule's full
  state_dict(), i.e. the whole 3B-parameter base model plus the LoRA adapter, not
  just the ~12M trainable LoRA weights. Only the single latest file is kept
  (like maestro's own checkpoint callback) to avoid piling up multi-GB files.
- configure_optimizers() here returns a plain AdamW with no LR scheduler, so
  there's no scheduler-state mismatch to worry about if you resume with a
  different GPU count / batch size than you stopped with; only Adam's per-param
  running averages, which continuing under a different effective batch size is a
  normal minor approximation, not a fatal problem.

Second bug found here, under real multi-GPU use (2026-09-23, 2-GPU DDP run,
crashed at epoch 37): maestro's own SaveCheckpoint.on_train_epoch_end
(maestro/trainer/common/callbacks.py) is NOT rank-zero-guarded --
`shutil.rmtree(checkpoint_path)` followed by recreating it runs on EVERY DDP
rank. With >1 GPU, all ranks race to delete+recreate the same directory (worse
on this project's NFS-backed output dir than it might be on local disk), and a
losing rank hits `OSError: [Errno 39] Directory not empty`, which is unhandled
and kills the whole DDP job (Lightning tears down all ranks together when one
crashes). Confirmed: epochs 33-36 saved fine, epoch 37 lost this race.

Fix: RankZeroSaveCheckpoint wraps it so only global rank 0 ever touches the
directory, matching how Lightning's own `trainer.save_checkpoint()` (used by
FullStateCheckpoint above) already handles rank safety internally -- that one
was never affected by this bug. Use RankZeroSaveCheckpoint in place of
maestro's SaveCheckpoint directly whenever training on >1 GPU.
"""
import lightning
from lightning.pytorch.callbacks import Callback
from maestro.trainer.common.callbacks import SaveCheckpoint


class FullStateCheckpoint(Callback):
    def __init__(self, path: str):
        self.path = path

    def on_train_epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        trainer.save_checkpoint(self.path)
        print(f"Saved full resume checkpoint (epoch {trainer.current_epoch}) to {self.path}")


class PeriodicAdapterSnapshot(Callback):
    """Keeps a deployable adapter snapshot every `every_n` epochs in
    <checkpoints>/epoch_XXX/ and NEVER overwrites it. maestro's own callback (and
    FullStateCheckpoint) only keep the LATEST epoch, so a run that peaks early and then
    overfits (loss ~0, worse held-out IoU) loses its best weights -- exactly what happened
    to the 100-epoch box run. ~50 MB per snapshot (LoRA adapter only), rank 0 only."""

    def __init__(self, result_path: str, every_n: int, save_model_callback):
        self.result_path = result_path
        self.every_n = every_n
        self.save_model_callback = save_model_callback

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1  # 1-based: "epoch_005" = weights after 5 finished epochs
        if trainer.is_global_zero and self.every_n > 0 and epoch % self.every_n == 0:
            path = f"{self.result_path}/epoch_{epoch:03d}"
            self.save_model_callback(path, pl_module.processor, pl_module.model)
            print(f"Saved permanent adapter snapshot (after {epoch} epochs) to {path}")


class RankZeroSaveCheckpoint(SaveCheckpoint):
    """maestro's SaveCheckpoint, but only ever executed on rank 0 under DDP --
    see the module docstring above for the race condition this avoids."""

    def on_train_epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        if trainer.is_global_zero:
            super().on_train_epoch_end(trainer, pl_module)
