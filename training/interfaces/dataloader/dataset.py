from collections.abc import Iterator, Sequence
import logging
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np

import training.models.model as _model
import training.configs.config as _config
import training.interfaces.transforms as _transforms
from torch.utils.data import ConcatDataset

T_co = TypeVar("T_co", covariant=True)

def _recursive_set_epoch(dataset, epoch: int):
    """辅助函数：递归向下探测并设置 epoch"""
    # 1. 尝试直接设置当前层
    if hasattr(dataset, "set_epoch") and callable(dataset.set_epoch):
        dataset.set_epoch(epoch)
    
    # 2. 如果是包装类（如 TransformedDataset），处理其内部 dataset
    inner_ds = getattr(dataset, "_dataset", None)
    if inner_ds is not None:
        _recursive_set_epoch(inner_ds, epoch)
        
    # 3. 如果是列表容器（如 WeightedMixtureDataset 的子集），遍历处理
    inner_datasets = getattr(dataset, "_datasets", None)
    if inner_datasets is not None:
        for ds in inner_datasets:
            _recursive_set_epoch(ds, epoch)

class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        try:
            return self._transform(self._dataset[index])
        except Exception as e:
            # 计算下一个索引，注意防止越界
            idx = int(index)
            next_index = (idx + 1) % len(self._dataset)
            print(f"Warning: Error at index {idx} in dataset {self._dataset}. Skipping to {next_index}. Error: {e}")
            
            # 递归调用。如果下一帧也坏了，它会继续找直到找到好的。
            return self.__getitem__(next_index)

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                batch_size = next(v.shape[0] for v in sample.values())
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]
                transformed = [self._transform(s) for s in individual_samples]
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples

class WeightedMixtureDataset(Dataset):
    def __init__(
        self,
        datasets: list[Dataset],
        weights: list[float],
        balance_by_length: bool = True,
        num_samples: int | None = None,
        seed: int = 42,
        name: str = "Mixture",
    ):
        self._datasets = datasets
        self._lengths = np.array([len(d) for d in datasets])

        def get_raw_len(ds):
            # 1. 如果是 TransformedDataset 或 MultiRepoLeRobotDataset 这种包装类
            # 必须先剥开它，否则会漏掉它内部可能含有的混合器
            for attr in ["_dataset", "dataset", "_combined_dataset"]:
                inner = getattr(ds, attr, None)
                if inner is not None:
                    return get_raw_len(inner)

            # 2. 如果当前层就是混合器，调用其专门的物理长度获取方法
            if hasattr(ds, "_get_physical_raw_len"):
                return ds._get_physical_raw_len()
            
            # 3. 实在没有包装也没有混合器了，返回当前的 len
            return len(ds)

        self._physical_lengths = np.array([get_raw_len(d) for d in datasets])

        self.seed = seed
        self.epoch = 0
        self.name = name

        raw_weights = np.array(weights, dtype=np.float32)
        max_w = raw_weights.max()
        if max_w <= 0:
            raise ValueError("Weights must contain at least one positive value.")
        self._raw_weights = raw_weights / max_w  # 归一化，确保最大值是 1.0

        # 核心逻辑：计算采样概率
        sampling_weights = self._raw_weights.copy()
    
        if balance_by_length:
            sampling_weights *= self._lengths
    
        # 归一化为概率分布
        self._sampling_probs = sampling_weights / sampling_weights.sum()

        # 计算总长度：参考主数据集（权重为1.0）

        if num_samples is None:
            primary_indices = self._raw_weights == 1.0
    
            self._len = int((self._lengths / self._sampling_probs)[primary_indices].max())
        else:
            self._len = num_samples

        if "Internal" in self.name:
            print(f"[{self.name}]")
            # 内部合并：关注物理数据到逻辑长度的压缩/扩张
            print(f"  - Total Physical Data: {self._physical_lengths.sum()} steps")
            print(f"  - Virtual Task Length: {self._len} (after weighting)")
            for i, (pl, p, w) in enumerate(zip(self._physical_lengths, self._sampling_probs, self._raw_weights)):
                 # 内部合并时 phys_len 和 reported_len 通常相等，只打印一个
                 print(f"    repo_{i}: len={pl}, weight={w:.2f}, prob={p:.4f}")
        
        else:
            print()
            print(f"[{self.name}]")
            # 异构合并：展示三级长度对比
            print(f"  - Real Physical Steps (on disk): {self._physical_lengths.sum()}") # 1. 最原始物理总长度
            print(f"  - Sum of Task Input Lengths: {self._lengths.sum()}")      # 2. 各自内部采样后的长度和
            print(f"  - Final Combined Epoch Length: {self._len}")          # 3. 混合后的最终长度
            print(f"  - Mixture strategy: balance_by_length={balance_by_length}")
            
            for i, (pl, rl, p, w) in enumerate(zip(self._physical_lengths, self._lengths, self._sampling_probs, self._raw_weights)):
                 # 此时 rl 是子任务汇报的逻辑长，pl 是该子任务对应的底层物理长
                 score = rl * w if balance_by_length else w
                 print(f"    task_{i}: input_len={rl}, phys_len={pl}, weight={w:.2f}, prob={p:.4f}, (score={score:.1f})")

    def _get_physical_raw_len(self):
        return self._physical_lengths.sum()

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self._len

    def __getitem__(self, index: SupportsIndex):
        idx = index.__index__()
        
        # 确定性种子序列，确保分布式环境下 Rank 间同步
        seed_sequence = [self.seed, self.epoch, idx,1024]
        rng = np.random.default_rng(seed_sequence)
        
        # 1. 选子数据集 (使用确定性 RNG)
        dataset_idx = rng.choice(len(self._datasets), p=self._sampling_probs)
        target_ds = self._datasets[dataset_idx]
        
        # 2. 在子数据集内随机选一个物理索引
        inner_idx = int(rng.integers(0, len(target_ds)))
        
        return target_ds[inner_idx]
    
class MultiRepoLeRobotDataset(Dataset):
    def __init__(
        self, 
        repo_ids: list[str], 
        data_config: _config.DataConfig, 
        action_horizon: int, 
        action_fps: int | None
    ):
        datasets = []
        # 假设 data_config.repo_weights 是一个 list[float]
        repo_weights = getattr(data_config, "repo_weights", [1.0] * len(repo_ids))

        for repo_id in repo_ids:
            meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
            
            # 修复：还原 Aloha 的特殊 FPS 处理逻辑
            curr_fps = action_fps
            # if 'aloha_lerobot' in repo_id:
            #     curr_fps = None
            
            sample_horizon = round(action_horizon * meta.fps / curr_fps) if curr_fps else action_horizon
            ds = lerobot_dataset.LeRobotDataset(
                repo_id,
                delta_timestamps={
                    key: [t / meta.fps for t in range(sample_horizon)] 
                    for key in data_config.action_sequence_keys
                },
            )
            if data_config.prompt_from_task:
                ds = TransformedDataset(ds, [_transforms.PromptFromLeRobotTask(meta.tasks)])
            datasets.append(ds)
            
        # 替换 ConcatDataset 为我们的加权采样类
        self._combined_dataset = WeightedMixtureDataset(
            datasets, 
            repo_weights, 
            balance_by_length=True, # 对应 balance_dataset_weights = True
            name="Task-Internal Repo Mixture"
        )

    def set_epoch(self, epoch: int):
        self._combined_dataset.set_epoch(epoch)

    def __getitem__(self, index: SupportsIndex):
        return self._combined_dataset[index]

    def __len__(self) -> int:
        return len(self._combined_dataset)


class DatasetGroup(Dataset):
    def __init__(self, config: _config.TrainConfig, skip_norm_stats: bool = False):
        factories = config.data_dict if config.data_dict is not None else {"default": config.data}
        # 假设 config.hetero_weights 是一个 dict[str, float]
        task_weights = []

        self._datasets = []
        self.data_configs = []

        for key, factory in factories.items():
            data_cfg = factory.create(config.assets_dirs, config.model)
            self.data_configs.append(data_cfg)
            
            # 修复：正确传递 horizon 和 fps 参数
            ds = create_torch_dataset(data_cfg, config.model.action_horizon, config.model.action_fps, config.model)
            ds = transform_dataset(ds, data_cfg, skip_norm_stats=skip_norm_stats)
            
            self._datasets.append(ds)
            weight = getattr(factory, "weight", 1.0)
            task_weights.append(float(weight))

        # 异构合并：开启长度平衡，确保大任务和小任务的每一帧概率均等（除非权重不同）
        self._dataset = WeightedMixtureDataset(self._datasets, task_weights, balance_by_length=True,name="Heterogeneous Task Mixture")

    @property
    def aggregated_config(self):
        # 返回第一个 config 供 DataLoaderImpl 使用（或可根据需要合并）
        return self.data_configs[0]

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        # 确保混合器更新
        _recursive_set_epoch(self._dataset, epoch)
        # 显式确保列表中的每个工厂产出的数据集都更新
        for ds in self._datasets:
            _recursive_set_epoch(ds, epoch)

    def __getitem__(self, index: SupportsIndex):
        return self._dataset[index]

    def __len__(self) -> int:
        return len(self._dataset)


def create_torch_dataset(
    data_config: _config.DataConfig, 
    action_horizon: int, 
    action_fps: int, 
    model_config: _model.BaseModelConfig
) -> Dataset:
    """创建基础 Dataset 实例（不包含 Transform）。"""
    # if data_config.repo_id == "fake":
    #     return FakeDataset(model_config, num_samples=1024)

    # 统一使用列表逻辑处理单库和同构多库
    repo_list = data_config.repo_id_list
    if not any(repo_list):
        raise ValueError("Neither repo_id nor repo_id_list is correctly set.")

    return MultiRepoLeRobotDataset(repo_list, data_config, action_horizon, action_fps)


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """对 Dataset 应用标准化转换流。"""
    norm_stats = {}
    if  not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError("Normalization stats not found. Run scripts/compute_norm_stats.py first.")
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """对 IterableDataset 应用标准化转换流。"""
    norm_stats = {}
    if  not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError("Normalization stats not found.")
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )
   