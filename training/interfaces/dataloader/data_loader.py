from collections.abc import Iterator
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, TypeVar

import jax
import numpy as np
import torch
from training.interfaces.dataloader.dataset import * # 假设之前的 DatasetGroup, create_torch_dataset 等在此定义

import training.models.model as _model
import training.configs.config as _config


T_co = TypeVar("T_co", covariant=True)

class KeyExtractorDataset(Dataset):
    def __init__(self, dataset: Dataset, keys: list[str] = ["state", "actions"]):
        self._dataset = dataset
        self._keys = keys

    def __getitem__(self, index: SupportsIndex) -> dict:
        data = self._dataset[index]
        # 只保留需要的键，这会瞬间释放掉图像占用的内存
        return {k: data[k] for k in self._keys if k in data}

    def __len__(self) -> int:
        return len(self._dataset)
    
def get_physical_datasets(dataset):
    """递归提取所有底层物理数据集，保留 Transform 但忽视采样权重"""
    # 1. 处理 DatasetGroup (异构层)
    if hasattr(dataset, "_datasets") and hasattr(dataset, "_dataset"):
        # 如果是 DatasetGroup，我们要它里面的列表
        leaves = []
        for d in dataset._datasets:
            leaves.extend(get_physical_datasets(d))
        return leaves

    # 2. 处理 MultiRepoLeRobotDataset (同构层)
    if hasattr(dataset, "_combined_dataset"):
        # 递归进入 WeightedMixtureDataset
        return get_physical_datasets(dataset._combined_dataset)

    # 3. 处理 WeightedMixtureDataset (采样器)
    if hasattr(dataset, "_datasets") and not hasattr(dataset, "_dataset"):
        leaves = []
        for d in dataset._datasets:
            leaves.extend(get_physical_datasets(d))
        return leaves

    # 4. 如果已经是叶子节点 (TransformedDataset 或 LeRobotDataset)
    return [KeyExtractorDataset(dataset, keys=["state", "actions"])]


class DataLoader(Protocol[T_co]):
    """数据加载器的通用接口。"""

    def data_config(self) -> _config.DataConfig:
        """获取当前数据加载器的配置信息。"""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        """迭代获取数据。"""
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")

def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """创建数据加载器的统一入口。
    
    支持单配置 (config.data) 和异构多配置 (config.data_dict)。
    """
    

    # 2. Torch 路由逻辑 (覆盖 LeRobot, Fake, 多库合并等)
    return create_torch_data_loader(
        config=config,
        action_horizon=config.model.action_horizon,
        action_fps=config.model.action_fps,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    config: _config.TrainConfig,
    action_horizon: int,
    action_fps: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """统一的 Torch 数据加载器。支持 JAX 和 PyTorch 的单机/多机环境。"""

    # 1. 探测分布式环境参数
    if framework == "pytorch":
        dist_world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        dist_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    else:
        dist_world_size = jax.process_count()
        dist_rank = jax.process_index() if dist_world_size > 1 else 0

    # 2. 构建数据集 (由 DatasetGroup 内部处理单/异构逻辑)
    dataset = DatasetGroup(config, skip_norm_stats=skip_norm_stats)

    # 3. 分布式采样器配置
    sampler = None
    if dist_world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist_world_size,
            rank=dist_rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )

    # 4. 计算本地 batch 大小
    local_batch_size = batch_size // dist_world_size
    print(f"Initialized {framework} DataLoader: total_batch={batch_size}, local_batch={local_batch_size}")
    
    # 5. 构建 PyTorch DataLoader
    generator = torch.Generator()
    generator.manual_seed(seed)
    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None

    torch_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=local_batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        collate_fn=_collate_fn,
        worker_init_fn=_worker_init_fn,
        # persistent_workers=num_workers > 0,
        persistent_workers=False,

        drop_last=True,
        generator=generator,
        multiprocessing_context=mp_context,
    )

    # 6. 封装底层加载器
    data_loader = TorchDataLoader(
        torch_loader,
        sharding=sharding,
        num_batches=num_batches,
        framework=framework,
    )

    return DataLoaderImpl(dataset.aggregated_config, data_loader)

class TorchDataLoader:
    """内部迭代器，负责将 PyTorch Batch 转换为框架特定的格式 (JAX Array 或 Torch Tensor)。"""
    def __init__(
        self,
        data_loader: torch.utils.data.DataLoader,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
        framework: str = "jax",
    ):
        self._data_loader = data_loader
        self._num_batches = num_batches
        self._framework = framework
        self._sharding = sharding
        
        if self._sharding is None and framework == "jax":
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

    def __iter__(self):
        num_items = 0
        current_epoch = 0
        
        while True:
            # 2. 同步 Sampler 的 epoch (用于分布式打乱)
            if hasattr(self._data_loader.sampler, "set_epoch"):
                self._data_loader.sampler.set_epoch(current_epoch)
            
            # 2. 设置 Dataset 的 epoch (关键：通过主进程实例设置)
            # 如果 persistent_workers=True, 需要通过 worker_init_fn 配合来同步，
            # 但在这里，简单的实现是每一轮手动更新主进程的 dataset 引用。
            if hasattr(self._data_loader.dataset, "set_epoch"):
                self._data_loader.dataset.set_epoch(current_epoch)

            data_iter = iter(self._data_loader)
            
            while True:
                # 检查是否达到用户指定的总 batch 数限制
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # 这一轮 epoch 跑完了
                    break
                
                num_items += 1
                if self._framework == "jax":
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)

            # 一轮 epoch 结束后，自增计数器
            logging.info(f"Epoch {current_epoch} completed")
            current_epoch += 1



class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: typing.Iterable):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]


def _collate_fn(items):
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)

def _worker_init_fn(worker_id: int) -> None:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"