import os
import random
import multiprocessing
from typing import Sequence, TypeVar, Any, Dict, List
from pathlib import Path

import numpy as np
import torch
import tqdm
import tyro
import jax
from torch.utils.data import ConcatDataset, DataLoader, Subset

import training.interfaces.shared.normalize as normalize
import training.configs.config as _config
import training.interfaces.transforms as _transforms
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

T_co = TypeVar("T_co", covariant=True)

class FilterKeys(_transforms.DataTransformFn):
    """只保留指定的键，从源头切断图像加载"""
    def __init__(self, keep_keys: List[str]):
        self.keep_keys = keep_keys

    def __call__(self, x: dict) -> dict:
        # 仅保留 state, actions 等必要的数值键
        return {k: v for k, v in x.items() if k in self.keep_keys}

class RemoveStrings(_transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}

class TransformedDataset:
    def __init__(self, dataset: Any, transforms_list: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms_list)

    def __getitem__(self, index: int) -> dict:
        try:
            return self._transform(self._dataset[index])
        except Exception as e:
            print(self._dataset[index])
            print(f"Error at index {index} in dataset {self._dataset}")
            raise

    def __len__(self) -> int:
        return len(self._dataset)

class NormDatasetManager:
    def __init__(self, config: _config.TrainConfig, sample_ratio: float = 0.001):
        self.config = config
        self.sample_ratio = sample_ratio
        self.model_cfg = config.model
        self.assets_dirs = config.assets_dirs
        # 预定义计算归一化需要的键
        self.keep_keys = ["state", "actions"]

    def _build_single_dataset(self, data_gen: Any) -> tuple[torch.utils.data.Dataset, _config.DataConfig]:
        data_config = data_gen.create(self.assets_dirs, self.model_cfg)
        repo_ids = data_config.repo_id_list if data_config.repo_id_list else [data_config.repo_id]
        
        sub_datasets = []
        for rid in repo_ids:
            meta = lerobot_dataset.LeRobotDatasetMetadata(rid)
            
            if self.model_cfg.action_fps is not None:
                sample_horizon = round(self.model_cfg.action_horizon * meta.fps / self.model_cfg.action_fps)
            else:
                sample_horizon = self.model_cfg.action_horizon

            ds = lerobot_dataset.LeRobotDataset(
                rid,
                delta_timestamps={
                    key: [t / meta.fps for t in range(sample_horizon)] 
                    for key in data_config.action_sequence_keys
                },
            )
            if data_config.prompt_from_task:
                ds = TransformedDataset(ds, [_transforms.PromptFromLeRobotTask(meta.tasks)])
            sub_datasets.append(ds)

        combined = ConcatDataset(sub_datasets) if len(sub_datasets) > 1 else sub_datasets[0]
        
        # 核心逻辑：先 FilterKeys 丢弃图像，再进行 DataTransform 处理
        # 这样 DataTransform 内部如果涉及图像处理也不会被执行，进一步加速
        final_ds = TransformedDataset(combined, [
            
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            FilterKeys(self.keep_keys), # 关键优化点：从这里开始图像数据就不存在了

            RemoveStrings()
        ])
        
        if self.sample_ratio < 1.0:
            subset_size = max(1, int(len(final_ds) * self.sample_ratio))
            indices = random.sample(range(len(final_ds)), subset_size)
            final_ds = Subset(final_ds, indices)
            
        return final_ds, data_config

    def get_dataloader(self, num_workers: int = 8) -> tuple[DataLoader, int, Path]:
        data_items = list(self.config.data_dict.items()) if self.config.data_dict else [("default", self.config.data)]
        
        all_datasets = []
        first_data_config = None
        
        for i, (_, data_gen) in enumerate(data_items):
            ds, data_cfg = self._build_single_dataset(data_gen)
            all_datasets.append(ds)
            if i == 0:
                first_data_config = data_cfg

        final_dataset = ConcatDataset(all_datasets) if len(all_datasets) > 1 else all_datasets[0]
        num_batches = max(1, len(final_dataset) // self.config.batch_size)
        
        loader = DataLoader(
            final_dataset,
            batch_size=self.config.batch_size,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            worker_init_fn=self._worker_init_fn,
            shuffle=True,
            drop_last=True,
            multiprocessing_context=multiprocessing.get_context("spawn") if num_workers > 0 else None
        )
        
        save_path = self.assets_dirs 
        return loader, num_batches, save_path

    @staticmethod
    def _collate_fn(items):
        return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

def main(
    config_name: str, 
    max_frames: int | None = None, 
    sample_ratio: float = 0.1,
    num_workers: int = 8
):
    config = _config.get_config(config_name)
    manager = NormDatasetManager(config, sample_ratio=sample_ratio)
    data_loader, total_batches, save_path = manager.get_dataloader(num_workers=num_workers)

    if max_frames:
        total_batches = min(total_batches, max_frames // config.batch_size)
    
    keys = ["state", "actions"]
    stats_accumulators = {k: normalize.RunningStats() for k in keys}

    print(f"Dataset samples after sampling: {len(data_loader.dataset)}")
    print(f"Stats will be computed only for keys: {keys}")

    for i, batch in enumerate(tqdm.tqdm(data_loader, total=total_batches, desc="Stats Progress")):
        if i >= total_batches:
            break
        for key in keys:
            if key in batch:
                stats_accumulators[key].update(np.asarray(batch[key]))

    norm_stats = {k: sa.get_statistics() for k, sa in stats_accumulators.items()}
    print(f"Writing stats to: {save_path}")
    normalize.save(save_path, norm_stats)

if __name__ == "__main__":
    tyro.cli(main)