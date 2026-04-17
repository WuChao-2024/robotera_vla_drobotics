"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias
from typing import List
import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro
import yaml
import numpy as np
import training.models.model as _model
import training.models.pi0_config as pi0_config
import training.models.tokenizer as _tokenizer
import training.interfaces.policies.m7_policy as m7_policy
import training.interfaces.shared.download as _download
import training.interfaces.shared.normalize as _normalize
import training.interfaces.misc.roboarena_config as roboarena_config
import training.utils.optimizer as _optimizer
import training.utils.weight_loaders as weight_loaders
import training.interfaces.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    assets_dir: str | None = None
    asset_id: str | None = None

@dataclasses.dataclass(frozen=True)
class DataConfig:
    # repo_id: str | None = None
    repo_id_list: List[str] | None = None
    repo_weights: List[float] | None = None
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | None = None
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    use_quantile_norm: bool = False
    action_sequence_keys: Sequence[str] = ("actions",)
    prompt_from_task: bool = False


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    weight: float = 1.0
    # repo_id: str = tyro.MISSING
    repo_id_list: List[str] = tyro.MISSING
    repo_weights: List[float] | None = None
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        repo_id_list = self.repo_id_list if self.repo_id_list is not tyro.MISSING else None
        # asset_id = self.assets.asset_id or repo_id
        eff_repo_weights = self.repo_weights
        if eff_repo_weights is None:
            count = len(self.repo_id_list) if self.repo_id_list else 1
            eff_repo_weights = [1.0] * count
        return dataclasses.replace(
            self.base_config or DataConfig(),
            # repo_id=repo_id,
            repo_id_list=repo_id_list,
            repo_weights=eff_repo_weights,
            # asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs)),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path) -> dict[str, _transforms.NormStats] | None:
        try:
            data_assets_dir = str(assets_dir)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    # repo_id: str = "fake"
    repo_id_list: List[str] = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id_list=self.repo_id_list)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )

  
@dataclasses.dataclass(frozen=True)
class LeRobotM7DataConfig(DataConfigFactory):
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    adapt_to_pi: bool = True
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(inputs=[_transforms.RepackTransform({})]))
    action_sequence_keys: Sequence[str] = ("",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[m7_policy.M7Inputs()],
            outputs=[m7_policy.M7Outputs()],
        )
        if self.use_delta_joint_actions:

            data_transforms = data_transforms.push(
                inputs=[_transforms.CamDeltaEeActions(model_config.action_horizon)], 
                outputs=[_transforms.CamAbsoluteEeActions()],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )




@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    exp_name: str = tyro.MISSING
    project_name: str = "robotera_vla"
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)
    pytorch_weight_path: str | None = None
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    data_dict: dict[str, Any] | None = None
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 16
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 10000  # yzy
    keep_period: int | None = 10000 # yzy
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True # yzy
    policy_metadata: dict[str, Any] | None = None
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")

def load_data_dict_from_yaml(yaml_path: str) -> dict[str, Any]:
    try:
        with open(yaml_path, 'r') as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file {yaml_path} not found.")
        return {}

    # 1. 动态获取所有子类
    def get_all_subclasses(cls):
        subs = []
        for s in cls.__subclasses__():
            subs.append(s)
            subs.extend(get_all_subclasses(s))
        return subs

    registry = {cls.__name__: cls for cls in get_all_subclasses(DataConfigFactory)}
    result = {}

    # 添加numpy数组转换函数
    def convert_to_numpy(obj):
        if isinstance(obj, list):
            # 检查是否是二维列表且所有行长度相同
            if obj and all(isinstance(row, list) for row in obj):
                row_lengths = [len(row) for row in obj]
                if len(set(row_lengths)) == 1:  # 所有行长度相同
                    return np.array(obj)
            # 普通列表
            return np.array(obj)
        return obj

    for k, v in raw.items():
        class_type = v.pop("type", None)
        cls = registry.get(class_type)
        if not cls:
            logging.warning(f"Unknown type {class_type} in YAML, skipping.")
            continue

        # 处理需要转换为numpy的字段
        numpy_fields = ['transform_mat_l', 'transform_mat_r']  # 可以根据需要添加更多字段
        for field in numpy_fields:
            if field in v and v[field] is not None:
                v[field] = convert_to_numpy(v[field])

        if "base_config" in v and isinstance(v["base_config"], dict):
            # 确保 list 转换为 tuple (action_sequence_keys)
            if "action_sequence_keys" in v["base_config"]:
                v["base_config"]["action_sequence_keys"] = tuple(v["base_config"]["action_sequence_keys"])
            v["base_config"] = DataConfig(**v["base_config"])
        else:
            v["base_config"] = DataConfig()

        if "repack_transforms" in v and isinstance(v["repack_transforms"], dict):
            # 将普通字典包装成框架需要的 Group -> RepackTransform 结构
            v["repack_transforms"] = _transforms.Group(
                inputs=[_transforms.RepackTransform(v["repack_transforms"])]
            )

        if "action_sequence_keys" in v and isinstance(v["action_sequence_keys"], list):
            v["action_sequence_keys"] = tuple(v["action_sequence_keys"])

        # 2. 自动过滤掉类构造函数不支持的参数 (防止 YAML 冗余字段报错)
        import inspect
        sig = inspect.signature(cls)
        valid_v = {param: v[param] for param in v if param in sig.parameters}
        
        # 3. 实例化对应的 ConfigFactory
        result[k] = cls(**valid_v)
        
    return result
_CONFIGS = [


    TrainConfig(
        name="pi05_M7_pp_opensource",

        #action_horizon:20, 是action_chunk
        model=pi0_config.Pi0Config(action_horizon=20,action_dim=38,pi05=True,no_state=True),
                                                                  # 默认配置为 gemma_2b(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256) 
                                                                  # gemma_300m(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
        data_dict=load_data_dict_from_yaml("training/configs/data_config_260322.yml"),
        batch_size=64, # 所有GPU上的样本数总和，而不是单个GPU上的批量大小 32（1.2s/it）-> 64（1.8s/it） —>128（3.6s/it）
        weight_loader=weight_loaders.NoActionWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        # weight_loader=weight_loaders.NoActionWeightLoader("/era-ai/lm/user/lhy/openpi/checkpoints/pi05_M7_good2people/260127/150000/params"),
        # weight_loader=weight_loaders.CheckpointWeightLoader("/era-ai/lm/weight/pi05/mooncake_chain_anything_1030/50000/params"),
        num_train_steps=300000,  # 300k -> 300k/2 -> 300k/4 = 75k
        fsdp_devices=8,  # 如果只有1张GPU，fsdp_devices=1
    ),


    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
