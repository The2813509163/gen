# src/llamafactory/data/custom_dataset.py

import itertools
from torch.utils.data import IterableDataset
from typing import List, Dict
from datasets import Dataset

class FixedRatioMixedDataset(IterableDataset):
    def __init__(self, sft_datasets: List[Dataset], dummy_dataset: Dataset, sft_ratio: int ,dummy_ratio: int):
        super().__init__()
        # 如果有多个SFT数据集，先把它们连接起来
        # 注意：这里假设sft_datasets列表中的数据集已经经过预处理
        self.sft_dataset = sft_datasets[0] # 简化处理，假设只有一个SFT源
        if len(sft_datasets) > 1:
            # 实际中可能需要更复杂的合并逻辑
            print("Warning: Multiple SFT datasets found for fixed mixing, only the first will be used in this simple example.")

        self.dummy_dataset = dummy_dataset
        self.sft_ratio = sft_ratio
        self.dummy_ratio = dummy_ratio

    def __iter__(self):
        # 创建可以无限循环的迭代器，并打乱数据
        sft_iter = itertools.cycle(self.sft_dataset)
        dummy_iter = itertools.cycle(self.dummy_dataset)

        while True:
            # 先 yield sft_ratio 个 SFT 样本
            for _ in range(self.sft_ratio):
                yield next(sft_iter)
            
            # 再 yield 1 个 PT 样本
            for _ in range(self.dummy_ratio):
                yield next(dummy_iter)