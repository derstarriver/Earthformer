# Find the original code and discussion at https://github.com/PyTorchLightning/pytorch-lightning/discussions/10922
# We will need to use the AMP implementation from apex because https://discuss.pytorch.org/t/using-torch-utils-checkpoint-checkpoint-with-dataparallel/78452

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.overrides.base import (
    _LightningModuleWrapperBase,
    _LightningPrecisionModuleWrapperBase,
)

try:
    from apex.parallel import DistributedDataParallel as ApexDDP
    _HAS_APEX = True
except ImportError:
    ApexDDP = None
    _HAS_APEX = False


def unwrap_lightning_module(wrapped_model):
    model = wrapped_model
    if ApexDDP is not None and isinstance(model, ApexDDP):
        model = unwrap_lightning_module(model.module)
    if isinstance(
        model, (_LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase)
    ):
        model = unwrap_lightning_module(model.module)
    return model


class ApexDDPStrategy(DDPStrategy):
    def __init__(self, *args, **kwargs):
        if not _HAS_APEX:
            raise ImportError(
                "ApexDDPStrategy requires NVIDIA Apex. Install it or use strategy='auto' for single GPU.")
        super().__init__(*args, **kwargs)

    def _setup_model(self, model):
        return ApexDDP(model, delay_allreduce=False)

    @property
    def lightning_module(self):
        return unwrap_lightning_module(self._model)


if __name__ == "__main__":
    # Correct usage of apex DDP, which can avoid error caused by using `torch.utils.checkpoint`
    # when using `strategy="ddp"` in pl.
    import pytorch_lightning as pl
    trainer = pl.Trainer(
        strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True),  # "ddp",
    )
