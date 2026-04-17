"""RoboArena baseline policy configs."""

from typing import TypeAlias

import training.models.model as _model
import training.models.pi0_config as pi0_config

import training.models.tokenizer as _tokenizer
import training.interfaces.policies.droid_policy as droid_policy
import training.interfaces.transforms as _transforms

ModelType: TypeAlias = _model.ModelType


def get_roboarena_configs():
    # Import here to avoid circular imports.
    from training.configs.config import AssetsConfig
    from training.configs.config import DataConfig
    from training.configs.config import SimpleDataConfig
    from training.configs.config import TrainConfig

    return [
        #
        # RoboArena DROID baseline inference configs.
        #
        TrainConfig(
            # pi0-style diffusion / flow VLA, trained on DROID from PaliGemma.
            name="paligemma_diffusion_droid",
            model=pi0_config.Pi0Config(action_horizon=10, action_dim=8),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
    ]
