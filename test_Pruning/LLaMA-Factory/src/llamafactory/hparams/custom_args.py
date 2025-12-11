# src/llmtuner/hparams/custom_args.py

from dataclasses import dataclass, field
from typing import Optional,List,Tuple

@dataclass
class CustomArguments:
    """
    """

    teacher_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "指向“教师模型”的路径，可用于知识蒸馏等任务。"}
    )

    dummy_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "指向“虚拟数据集”的路径，用于自定义的数据处理逻辑。"}
    )

    dummy_batch_size: int = field(
        default=0,
        metadata={"help": "处理“虚拟数据集”时每个设备的批处理大小 (batch size)。"}
    )

    alpha_schedule: Optional[List[Tuple[float, float]]] = field(
        default_factory=list,
        metadata={
            "help": (
                "一个动态调整 alpha 值的计划。格式为一系列的 [step_multiplier, alpha_value] 对。"
                "例如: [[1.2, 0.0005], [2.5, 0.0]]"
            )
        }
    )
