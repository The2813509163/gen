# Pruning-Aware Tuning v2 (PATv2)
[PAT](https://github.com/kriskrisliu/PAT)的升级版，在原有基础上做了多方面扩展形成PATv2，以下是对比：

| Feature                     | PAT                                   | PATv2                                     |
|-----------------------------|---------------------------------------|------------------------------------------|
| Hidden dimension pruning    | Yes                                   | Yes                                      |
| Intermediate dimension pruning | No                                    | Yes                                      |
| Learnable mask               | Yes, but with unstable gradients      | Yes, with enhanced gradient stability    |
| Training loss                | Static pruning loss                  | Ascending pruning loss, improves stability |
| Codebase                     | FireFly (no longer maintained)        | Llama-Factory (most widely used)         |

## checklist
- [X] 支持llama2、3系列
- [ ] 支持qwen3
- [ ] 如何下载和配置训练数据
- [ ] 如何配置evaluation数据
- [ ] 在json训练配置里面，需要注意哪些训练参数

## 0. 环境准备
准备4个环境：`factory-pat`, `factory-official`, `opencompass-pat`, `opencompass-official`
- `factory-pat`：PATv2的训练环境
- `factory-official`：llama-factory官方的训练环境
- `opencompass-pat`：PATv2的推理环境
- `opencompass-official`：opencompass官方的推理环境

```bash
# PAT 定制 llama-factory
conda create -n factory-pat python=3.10 -y
conda activate factory-pat
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e thirdparty/transformers-4.51.1
pip install -e thirdparty/peft-0.15.1
pip install tensorboard
pip install -e LLaMA-Factory

# PAT 定制 opencompass
conda create -n opencompass-pat python=3.10 -y
conda activate opencompass-pat
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install opencompass
pip install -e thirdparty/transformers-4.51.1
pip install -e thirdparty/peft-0.15.1

# 官方 llama-factory
conda create -n factory-official python=3.10 -y
conda activate factory-official
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.51.1
pip install peft==0.15.1
pip install tensorboard
pip install -e LLaMA-Factory-main

# 官方 opencompass
conda create -n opencompass-official python=3.10 -y
conda activate opencompass-official
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install opencompass
pip install transformers==4.51.1
pip install peft==0.15.1
```

## 1. 数据准备
- 修改 `LLaMA-Factory/data/dataset_info.json`
- 例如：`"hf_hub_url": "/data/ubuntu/kris/checkpoints/nvidia__OpenMathInstruct-2",`

## 2. 训练
剪枝和普通训练是独立的环境和代码，注意分开。
- 普通训练
```bash
conda activate factory-official
cd LLaMA-Factory-main
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_lora/your-yaml-file
```
- 剪枝训练
```bash
conda activate factory-pat
cd LLaMA-Factory
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_lora/your-yaml-file
```

## 3. 评估
普通模型和剪枝模型的评估是分开的。
### 普通
- 参考官方文档或help选项：`opencompass --help`
```bash
conda activate opencompass-official

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 opencompass --hf-type chat \
--hf-path /data/ubuntu/kris/checkpoints/meta-llama__Llama-3.2-3B-Instruct \
--peft-path your-trained-lora-path \
--datasets gsm8k_0shot_v2_gen_17d799  math_0shot_gen_11c4b5  svamp_gen_fb25e4  \
--batch-size 16 --max-out-len 512 --max-num-workers 8
```
### 剪枝
- 配置剪枝的环境变量： `tap_rank=512 tap_stop_at_steps=35000 update_step=78111`
- 其他与官方类似
- 无法启用vllm加速
```bash
conda activate opencompass-pat

tap_rank=512 tap_stop_at_steps=35000 update_step=78111 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 opencompass --hf-type chat \
--hf-path /data/ubuntu/kris/checkpoints/meta-llama__Llama-3.2-3B-Instruct \
--peft-path /data/ubuntu/kris/liuyijiang/meta-llama__Llama-3.2-3B-Instruct-tap0.5-learnable-interm-data2M-2epoch-hio512-simpleMLP/lora/sft/ \
--datasets gsm8k_0shot_v2_gen_17d799  math_0shot_gen_11c4b5  svamp_gen_fb25e4  \
--batch-size 16 --max-out-len 512 --max-num-workers 8
```

## 4. Pruning 代码核心改动部分
```bash
cat diff_output.txt
```
or
<details>
<summary>点击这里查看详细代码</summary>

```diff


diff --git a/LLaMA-Factory/data/dataset_info.json b/LLaMA-Factory/data/dataset_info.json
index 25e4029..47e9ba2 100644
--- a/LLaMA-Factory/data/dataset_info.json
+++ b/LLaMA-Factory/data/dataset_info.json
@@ -1,12 +1,13 @@
 {
   "nvidia__OpenMathInstruct-2":{
-    "hf_hub_url": "/home/kris/workspace/checkpoints/datasets/nvidia__OpenMathInstruct-2",
+    "hf_hub_url": "/root/krisliu/checkpoints/nvidia__OpenMathInstruct-2",
     "formatting": "alpaca",
     "columns":{
       "prompt": "problem",
       "query": null,
       "response": "generated_solution"  
-    }
+    },
+    "split": "train_5M"
   },
   "identity": {
     "file_name": "identity.json"


diff --git a/LLaMA-Factory/merging_lora_model.py b/LLaMA-Factory/merging_lora_model.py
index 29fc835..d44d75c 100644
--- a/LLaMA-Factory/merging_lora_model.py
+++ b/LLaMA-Factory/merging_lora_model.py
@@ -2,12 +2,24 @@ import torch
 from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
 from peft import PeftModel, PeftConfig
 import argparse
+import os
+from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer, LlamaMLP
+
 parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model and save merged model.")
 parser.add_argument("--base_model", type=str, required=True, help="Path or name of the base model.")
 parser.add_argument("--lora_model", type=str, required=True, help="Path to the LoRA adapter.")
 parser.add_argument("--output_model", type=str, required=True, help="Path to save the merged model.")
+parser.add_argument("--HIO_enabled", action="store_true", help="Enable HIO.")
+parser.add_argument("--HIO_r", type=int, default=None, help="HIO rank.")
+parser.add_argument("--remain_ratio", type=float, required=True, help="Remain ratio.")
+parser.add_argument("--learnable_mask", action="store_true", help="Use learnable mask.")
 args = parser.parse_args()
 
+if args.HIO_enabled and args.HIO_r is not None:
+    os.environ["HIO_r"] = str(args.HIO_r)
+if args.learnable_mask:
+    os.environ["learnable_mask"] = "true"
+
 
 # 1. Load the Base Model and the LoRA Adapter
 
@@ -34,13 +46,13 @@ try:
 
     )
     model.eval() # Set to evaluation mode
-    # Load the LoRA configuration
-    lora_config = PeftConfig.from_pretrained(lora_model_name_or_path)
 
-
-    # Load the LoRA adapter
-    model = PeftModel.from_pretrained(model, lora_model_name_or_path)
-    model.eval()
+    if args.lora_model != "skip":
+        # Load the LoRA configuration
+        lora_config = PeftConfig.from_pretrained(lora_model_name_or_path)
+        # Load the LoRA adapter
+        model = PeftModel.from_pretrained(model, lora_model_name_or_path)
+        model.eval()
 
     tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
     if tokenizer.pad_token is None:
@@ -51,11 +63,49 @@ except Exception as e:
     raise
 
 # 2.  Merge LoRA into the Base Model
+if args.lora_model != "skip":
+    tap_args = {
+        "tap_enabled": True,
+        "tap_rank": 512,
+        "tap_stop_at_steps": 6000,
+        'tap_remain_ratio': 0.5,
+        "tap_learnable_mask": True
+    }
+    setattr(model, "tap_args", tap_args)
+    for n,m in model.named_modules():
+        setattr(m, "update_step", 10000)
+        setattr(m, "tap_stop_at_steps", tap_args["tap_stop_at_steps"])
+        if isinstance(m, LlamaDecoderLayer) or isinstance(m, LlamaMLP):
+            setattr(m, "tap_args", tap_args)
+
+messages = [
+    {"role": "system", "content": "You are a helpful assistant!"},
+    {"role": "user", "content": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nPlease reason step by step, and put your final answer within \\boxed{}.\n"},
+]
+
+input_ids = tokenizer.apply_chat_template(
+    messages,
+    # add_generation_prompt=True,
+    return_tensors="pt"
+).to(model.device)
+
+model.to("cuda")
+with torch.no_grad():
+    model = model.merge_and_unload()
+    output = model.generate(input_ids, max_new_tokens=100)
+    print(tokenizer.decode(output[0], skip_special_tokens=True))
+raise NotImplementedError
+
 
 try:
     # Disable gradients for efficient merging
     with torch.no_grad():
         model = model.merge_and_unload()
+        for n, m in model.named_modules():
+            if isinstance(m, LlamaModel):
+                print(f"Merging HIO for {n}")
+                m.merge_HIO(remain_ratio=args.remain_ratio, learnable_mask=args.learnable_mask)
+                m.delete_HIO()
 except Exception as e:
     print(f"Error merging LoRA: {e}")
     raise


diff --git a/LLaMA-Factory/src/llamafactory/cli.py b/LLaMA-Factory/src/llamafactory/cli.py
index 075ef84..d4bd093 100644
--- a/LLaMA-Factory/src/llamafactory/cli.py
+++ b/LLaMA-Factory/src/llamafactory/cli.py
@@ -26,6 +26,8 @@ from .extras.env import VERSION, print_env
 from .extras.misc import find_available_port, get_device_count, is_env_enabled, use_ray
 from .train.tuner import export_model, run_exp
 from .webui.interface import run_web_demo, run_web_ui
+import yaml
+import shutil
 
 
 USAGE = (
@@ -74,6 +76,17 @@ class Command(str, Enum):
 
 def main():
     command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
+
+    yaml_path = sys.argv[-1]
+    with open(yaml_path, 'r') as file:
+        # Load the content of the YAML file
+        data = yaml.safe_load(file)
+    output_dir = data['output_dir']
+    # copy yaml file to output_dir
+    os.makedirs(output_dir, exist_ok=True)
+    shutil.copy(yaml_path, output_dir)
+
+
     if command == Command.API:
         run_api()
     elif command == Command.CHAT:


diff --git a/LLaMA-Factory/src/llamafactory/model/adapter.py b/LLaMA-Factory/src/llamafactory/model/adapter.py
index bbc0056..a610fef 100644
--- a/LLaMA-Factory/src/llamafactory/model/adapter.py
+++ b/LLaMA-Factory/src/llamafactory/model/adapter.py
@@ -25,6 +25,7 @@ from .model_utils.quantization import QuantizationMethod
 from .model_utils.unsloth import get_unsloth_peft_model, load_unsloth_peft_model
 from .model_utils.visual import get_forbidden_modules, patch_target_modules
 
+import math
 
 if TYPE_CHECKING:
     from transformers import PretrainedConfig, PreTrainedModel
@@ -248,6 +249,17 @@ def _setup_lora_tuning(
             model = get_peft_model(model, lora_config)
 
     if is_trainable and cast_trainable_params_to_fp32:
+        # TAP
+        for n,p in model.named_parameters():
+            if "HIO" in n:
+                p.requires_grad = True
+                if "HIO_A" in n:
+                    torch.nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))
+                elif "HIO_B" in n:
+                    torch.nn.init.zeros_(p.data)
+            if "HSM_mask_proxy" in n and (getattr(model.base_model.model.model, "learnable_mask", "False").lower()=='true'):
+                p.requires_grad = True
+                torch.nn.init.zeros_(p.data)
         for param in filter(lambda p: p.requires_grad, model.parameters()):
             param.data = param.data.to(torch.float32)
 


diff --git a/LLaMA-Factory/src/llamafactory/train/sft/workflow.py b/LLaMA-Factory/src/llamafactory/train/sft/workflow.py
index cf6f80a..6b8795f 100644
--- a/LLaMA-Factory/src/llamafactory/train/sft/workflow.py
+++ b/LLaMA-Factory/src/llamafactory/train/sft/workflow.py
@@ -26,7 +26,7 @@ from ...model import load_model, load_tokenizer
 from ..trainer_utils import create_modelcard_and_push
 from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
 from .trainer import CustomSeq2SeqTrainer
-
+from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaMLP
 
 if TYPE_CHECKING:
     from transformers import Seq2SeqTrainingArguments, TrainerCallback
@@ -44,13 +44,18 @@ def run_sft(
     finetuning_args: "FinetuningArguments",
     generating_args: "GeneratingArguments",
     callbacks: Optional[list["TrainerCallback"]] = None,
+    tap_args: Optional[dict] = None,
 ):
     tokenizer_module = load_tokenizer(model_args)
     tokenizer = tokenizer_module["tokenizer"]
     template = get_template_and_fix_tokenizer(tokenizer, data_args)
     dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
     model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
-
+    # TAP
+    setattr(model, "tap_args", tap_args)
+    for n,m in model.named_modules():
+        if isinstance(m, LlamaDecoderLayer) or isinstance(m, LlamaMLP):
+            setattr(m, "tap_args", tap_args)
     if getattr(model, "is_quantized", False) and not training_args.do_train:
         setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction
 


diff --git a/LLaMA-Factory/src/llamafactory/train/tuner.py b/LLaMA-Factory/src/llamafactory/train/tuner.py
index 3adb382..ff03f96 100644
--- a/LLaMA-Factory/src/llamafactory/train/tuner.py
+++ b/LLaMA-Factory/src/llamafactory/train/tuner.py
@@ -51,6 +51,19 @@ logger = logging.get_logger(__name__)
 
 def _training_function(config: dict[str, Any]) -> None:
     args = config.get("args")
+
+    # TAP
+    tap_args = dict(
+        tap_enabled = args.pop("tap_enabled", False),
+        tap_rank = args.pop("tap_rank", 512),
+        tap_stop_at_steps = args.pop("tap_stop_at_steps", 10000),
+        tap_remain_ratio = args.pop("tap_remain_ratio", 0.8),
+        tap_learnable_mask = args.pop("tap_learnable_mask", False),
+    )
+    if tap_args["tap_enabled"]:
+        os.environ["HIO_r"] = str(tap_args["tap_rank"])
+        os.environ["learnable_mask"] = str(tap_args["tap_learnable_mask"])
+
     callbacks: list[Any] = config.get("callbacks")
     model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
 
@@ -66,7 +79,7 @@ def _training_function(config: dict[str, Any]) -> None:
     if finetuning_args.stage == "pt":
         run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
     elif finetuning_args.stage == "sft":
-        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
+        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks, tap_args)
     elif finetuning_args.stage == "rm":
         run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
     elif finetuning_args.stage == "ppo":


diff --git a/thirdparty/peft-0.15.1/src/peft/utils/save_and_load.py b/thirdparty/peft-0.15.1/src/peft/utils/save_and_load.py
index f950b91..3dabf10 100644
--- a/thirdparty/peft-0.15.1/src/peft/utils/save_and_load.py
+++ b/thirdparty/peft-0.15.1/src/peft/utils/save_and_load.py
@@ -266,6 +266,9 @@ def get_peft_model_state_dict(
 
     # REMOVE ADAPTER NAME
     to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
+    for k, v in state_dict.items():
+        if "HIO" in k or "HSM_mask_proxy" in k:
+            to_return[k] = v
     return to_return
 
 


diff --git a/thirdparty/transformers-4.51.1/src/transformers/modeling_outputs.py b/thirdparty/transformers-4.51.1/src/transformers/modeling_outputs.py
index 60a3642..f27006e 100755
--- a/thirdparty/transformers-4.51.1/src/transformers/modeling_outputs.py
+++ b/thirdparty/transformers-4.51.1/src/transformers/modeling_outputs.py
@@ -157,6 +157,7 @@ class BaseModelOutputWithPast(ModelOutput):
     past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
     hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
     attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
+    pruning_interm_loss: Optional[torch.FloatTensor] = None
 
 
 @dataclass


diff --git a/thirdparty/transformers-4.51.1/src/transformers/models/llama/modeling_llama.py b/thirdparty/transformers-4.51.1/src/transformers/models/llama/modeling_llama.py
index 938fd7d..65578be 100644
--- a/thirdparty/transformers-4.51.1/src/transformers/models/llama/modeling_llama.py
+++ b/thirdparty/transformers-4.51.1/src/transformers/models/llama/modeling_llama.py
@@ -59,12 +59,38 @@ if is_torch_flex_attn_available():
 
     from ...integrations.flex_attention import make_flex_block_causal_mask
 
+import math, os
 
 logger = logging.get_logger(__name__)
 
 _CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
 _CONFIG_FOR_DOC = "LlamaConfig"
 
+def ascend_by_step(step, total_steps, begin_value, end_value):
+    return min(begin_value + (end_value - begin_value) * step / total_steps, end_value)
+
+
+class BinarizeFunction(torch.autograd.Function):
+    @staticmethod
+    def forward(ctx, input):
+        # Save input for backward pass, if necessary
+        ctx.save_for_backward(input)
+        
+        # Forward pass: convert input to 0 or 1 based on condition
+        return (input >= 0).float().to(input)  # .float() converts boolean to float (0.0 or 1.0)
+
+    @staticmethod
+    def backward(ctx, grad_output):
+        # Retrieve saved input, if necessary (not used here)
+        input, = ctx.saved_tensors
+        
+        # Straight through estimator: pass gradient through unchanged
+        grad_input = grad_output.clone()
+        
+        return grad_input
+
+# Alias for easier usage
+binarize = BinarizeFunction.apply
 
 class LlamaRMSNorm(nn.Module):
     def __init__(self, hidden_size, eps=1e-6):
@@ -168,9 +194,69 @@ class LlamaMLP(nn.Module):
         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
         self.act_fn = ACT2FN[config.hidden_act]
 
+    def init_HIO_for_MLP_layer(self, learnable_mask, hio_rank):
+        self.MLP_HIO_Version = ['complex', 'simple'][1]
+        if self.MLP_HIO_Version == 'complex':
+            self.HIO_A_gate_proj = nn.Parameter(torch.zeros(hio_rank, self.intermediate_size))
+            self.HIO_B_gate_proj = nn.Parameter(torch.zeros(self.intermediate_size, hio_rank))
+            self.HIO_A_up_proj = nn.Parameter(torch.zeros(hio_rank, self.intermediate_size))
+            self.HIO_B_up_proj = nn.Parameter(torch.zeros(self.intermediate_size, hio_rank))
+        self.HIO_A_down_proj = nn.Parameter(torch.zeros(hio_rank, self.intermediate_size))
+        self.HIO_B_down_proj = nn.Parameter(torch.zeros(self.intermediate_size, hio_rank))
+        self.learnable_mask = learnable_mask
+        if learnable_mask:
+            self.HSM_mask_proxy = nn.Parameter(torch.zeros(self.intermediate_size))
+        else:
+            raise NotImplementedError("Only learnable mask is supported for MLP")
+            self.HSM_mask_proxy = None
+
     def forward(self, x):
-        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
-        return down_proj
+        pruning_interm_loss = 0
+        if self.learnable_mask is not None:
+            if self.learnable_mask:
+                Mask = learnable_mask_dynamic_mapping_function(self.update_step, self.tap_stop_at_steps, self.HSM_mask_proxy, 0.8).squeeze()
+                if self.MLP_HIO_Version == 'complex':
+                    gate = self.gate_proj(x)
+                    gate = self.act_fn(gate + gate @ self.HIO_A_gate_proj.T @ self.HIO_B_gate_proj.T)
+                    up = self.up_proj(x)
+                    up = up + up @ self.HIO_A_up_proj.T @ self.HIO_B_up_proj.T
+                    down_proj = (gate * Mask.unsqueeze(0).unsqueeze(0) * up)
+                    down_proj = self.down_proj(down_proj + (down_proj @ self.HIO_A_down_proj.T @ self.HIO_B_down_proj.T) * Mask.unsqueeze(0).unsqueeze(0))
+                elif self.MLP_HIO_Version == 'simple':
+                    gate = self.act_fn(self.gate_proj(x))
+                    up = self.up_proj(x)
+                    down_proj = gate * up
+                    down_proj = self.down_proj(
+                        (
+                            down_proj + (down_proj @ self.HIO_A_down_proj.T @ self.HIO_B_down_proj.T)
+                        ) * Mask.unsqueeze(0).unsqueeze(0)
+                    )
+
+                
+                if self.update_step > 0.02* self.tap_args["tap_stop_at_steps"]:
+                    tap_remain_ratio = self.tap_args["tap_remain_ratio"]
+                    remain_dim_num = int(self.intermediate_size * tap_remain_ratio // 32 * 32)
+                    pruning_interm_loss = (
+                        10 * torch.abs(
+                            binarize(self.HSM_mask_proxy).sum() - remain_dim_num
+                        )/remain_dim_num
+                    )
+                    
+                    pruning_l2_loss = ascend_by_step(
+                        self.update_step, 1.5*self.tap_args["tap_stop_at_steps"], 0, 20
+                    ) * (self.HSM_mask_proxy[self.HSM_mask_proxy<=0.0] + 0.5).pow(2).mean()
+                        
+                    pruning_l1_loss = ascend_by_step(
+                        self.update_step, 1.5*self.tap_args["tap_stop_at_steps"], 0, 2
+                    ) * (self.HSM_mask_proxy[self.HSM_mask_proxy<=0.0] + 0.5).abs().mean()
+                    
+                    pruning_interm_loss = pruning_interm_loss + pruning_l2_loss + pruning_l1_loss
+
+            else:
+                raise NotImplementedError("Only learnable mask is supported for MLP")
+        else:
+            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
+        return down_proj, pruning_interm_loss
 
 
 def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
@@ -286,6 +372,133 @@ class LlamaAttention(nn.Module):
         attn_output = self.o_proj(attn_output)
         return attn_output, attn_weights
 
+def merge_HIO_general(self, type=None, remain_ratio=None, learnable_mask=False, mask_proxy=None):
+    if learnable_mask:
+        update_step = 10000
+        tap_stop_at_steps = 6000
+        early_stop_ratio = 0.8
+        mask = learnable_mask_dynamic_mapping_function(update_step, tap_stop_at_steps, mask_proxy, early_stop_ratio).squeeze()
+    else:
+        remain_dim_num = int(self.hidden_size * remain_ratio // 32 * 32)
+        mask = torch.ones(self.hidden_size)
+        mask[remain_dim_num:] = 0
+
+    if type == "decoder":
+
+        o_proj = self.HIO_B_Attn.cuda() @ self.HIO_A_Attn.cuda() @ self.self_attn.o_proj.weight.cuda() + self.self_attn.o_proj.weight.cuda()
+        o_proj = o_proj.cpu() * mask.unsqueeze(1)
+        self.self_attn.o_proj.weight.copy_(o_proj.detach().clone())
+        
+        down_proj = self.HIO_B_MLP.cuda() @ self.HIO_A_MLP.cuda() @ self.mlp.down_proj.weight.cuda() + self.mlp.down_proj.weight.cuda()
+        down_proj = down_proj.cpu() * mask.unsqueeze(1)
+        self.mlp.down_proj.weight.copy_(down_proj.detach().clone())
+        
+        if learnable_mask:
+            mask_copy = torch.ones_like(mask)
+            mask_copy[mask < 1e-5] = 0
+            mask = mask_copy
+
+        self.self_attn.q_proj.weight.copy_(self.self_attn.q_proj.weight.detach().clone()*mask.unsqueeze(0))
+        self.self_attn.k_proj.weight.copy_(self.self_attn.k_proj.weight.detach().clone()*mask.unsqueeze(0))
+        self.self_attn.v_proj.weight.copy_(self.self_attn.v_proj.weight.detach().clone()*mask.unsqueeze(0))
+
+        self.mlp.gate_proj.weight.copy_(self.mlp.gate_proj.weight.detach().clone()*mask.unsqueeze(0))
+        self.mlp.up_proj.weight.copy_(self.mlp.up_proj.weight.detach().clone()*mask.unsqueeze(0))
+    elif type == "embedding":
+        embed_tokens = self.embed_tokens.weight.cuda() + self.embed_tokens.weight.cuda() @ self.HIO_A_Emb.cuda().T @ self.HIO_B_Emb.cuda().T
+        embed_tokens = embed_tokens * mask.unsqueeze(0).cuda()
+        print("Zero values in mask:" ,mask[mask<1e-5].shape)
+        self.embed_tokens.weight.copy_(embed_tokens.detach().clone().cpu())
+    elif type == "mlp":
+
+        I_mask = torch.diag(mask).to("cuda", self.mlp.down_proj.weight.dtype)
+        down_proj = ((
+            I_mask + I_mask @ self.mlp.HIO_A_down_proj.T.cuda() @ self.mlp.HIO_B_down_proj.T.cuda() @ I_mask
+        ) @ self.mlp.down_proj.weight.cuda().T ).T
+        down_proj = down_proj.cpu()
+        self.mlp.down_proj.weight.copy_(down_proj.detach().clone())
+
+        I_mask_pure = torch.eye(self.mlp.intermediate_size).to("cuda", self.mlp.down_proj.weight.dtype) * (I_mask>0)
+        # gate_proj = I_mask_pure @ (
+        #     torch.eye(self.mlp.intermediate_size).to("cuda", self.mlp.gate_proj.weight.dtype) + self.mlp.HIO_A_gate_proj.T.cuda() @ self.mlp.HIO_B_gate_proj.T.cuda()
+        # ).T @ self.mlp.gate_proj.weight.cuda()
+
+        gate_proj = (
+            self.mlp.gate_proj.weight.cuda().T @ (
+                torch.eye(self.mlp.intermediate_size).to("cuda", self.mlp.gate_proj.weight.dtype) 
+                + self.mlp.HIO_A_gate_proj.T.cuda() @ self.mlp.HIO_B_gate_proj.T.cuda()
+            ) @ I_mask_pure
+        ).T
+
+        gate_proj = gate_proj.cpu()
+        self.mlp.gate_proj.weight.copy_(gate_proj.detach().clone())
+
+        # up_proj = I_mask_pure @ (
+        #     torch.eye(self.mlp.intermediate_size).to("cuda", self.mlp.up_proj.weight.dtype) + self.mlp.HIO_A_up_proj.T.cuda() @ self.mlp.HIO_B_up_proj.T.cuda()
+        # ).T @ self.mlp.up_proj.weight.cuda()
+
+        up_proj = (
+            self.mlp.up_proj.weight.cuda().T @ (
+                torch.eye(self.mlp.intermediate_size).to("cuda", self.mlp.up_proj.weight.dtype) 
+                + self.mlp.HIO_A_up_proj.T.cuda() @ self.mlp.HIO_B_up_proj.T.cuda()
+            ) @ I_mask_pure
+        ).T
+
+        up_proj = up_proj.cpu()
+        self.mlp.up_proj.weight.copy_(up_proj.detach().clone())
+
+def delete_HIO_general(self, type=None):
+    if type == "decoder":
+        del self.HIO_A_Attn
+        del self.HIO_B_Attn
+        del self.HIO_A_MLP
+        del self.HIO_B_MLP
+    elif type == "embedding":
+        del self.HIO_A_Emb
+        del self.HIO_B_Emb
+    elif type == "mlp":
+        del self.mlp.HIO_A_gate_proj
+        del self.mlp.HIO_B_gate_proj
+        del self.mlp.HIO_A_up_proj
+        del self.mlp.HIO_B_up_proj
+        del self.mlp.HIO_A_down_proj
+        del self.mlp.HIO_B_down_proj
+        del self.mlp.HSM_mask_proxy
+def learnable_mask_dynamic_mapping_function(update_step, tap_stop_at_steps, mask_proxy, early_stop_ratio=1.0):
+    update_step = torch.tensor(update_step)
+    tap_stop_at_steps = torch.tensor(tap_stop_at_steps)
+
+    beta = max(0.5 - update_step/tap_stop_at_steps, 0.0)
+    #### old
+    if update_step < tap_stop_at_steps * early_stop_ratio:
+        gamma = 1.0 / (
+            1.0 - torch.log(update_step)/torch.log(tap_stop_at_steps) + 1e-6
+        )
+        return (1 / (1 + torch.exp(- gamma * mask_proxy)) + beta)
+    else:
+        gamma = 1.0 / (
+            1.0 - torch.log(tap_stop_at_steps * early_stop_ratio)/torch.log(tap_stop_at_steps) + 1e-6
+        )
+        return (1 / (1 + torch.exp(- gamma * mask_proxy.detach())) + beta)
+    # return (
+    #         1 / (1 + torch.exp(- gamma * mask_proxy)) + beta
+    # )
+
+    #### new
+    # gamma = 1.0 / (
+    #         1.0 - torch.log(min(update_step, tap_stop_at_steps))/torch.log(tap_stop_at_steps) + 1e-6
+    #     )
+
+    # beta = max(0.5 - update_step/tap_stop_at_steps, 0.0)
+    
+    # if update_step < tap_stop_at_steps * early_stop_ratio:
+    #     return (
+    #         1 / (1 + torch.exp(- gamma * mask_proxy)) + beta
+    #     )
+    # else:
+    #     return (
+    #         1 / (1 + torch.exp(- gamma * mask_proxy.detach())) + beta
+    #     )
 
 class LlamaDecoderLayer(nn.Module):
     def __init__(self, config: LlamaConfig, layer_idx: int):
@@ -298,6 +511,24 @@ class LlamaDecoderLayer(nn.Module):
         self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
         self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
 
+        # TAP
+        HIO_r = int(os.getenv("HIO_r", 0))
+        if HIO_r > 0:
+            self.HIO_A_Attn = nn.Parameter(torch.zeros(HIO_r, config.hidden_size))
+            self.HIO_B_Attn = nn.Parameter(torch.zeros(config.hidden_size, HIO_r))
+            self.HIO_A_MLP = nn.Parameter(torch.zeros(HIO_r, config.hidden_size))
+            self.HIO_B_MLP = nn.Parameter(torch.zeros(config.hidden_size, HIO_r))
+            self.remain_dim_num = None
+            self.HSM_mask_proxy = torch.ones(1, 1, config.hidden_size)
+
+    def merge_HIO(self, remain_ratio, learnable_mask=False, mask_proxy=None):
+        merge_HIO_general(self, type="mlp", remain_ratio=remain_ratio, learnable_mask=learnable_mask, mask_proxy=self.mlp.HSM_mask_proxy)
+        merge_HIO_general(self, type="decoder", remain_ratio=remain_ratio, learnable_mask=learnable_mask, mask_proxy=mask_proxy)
+
+    def delete_HIO(self):
+        delete_HIO_general(self, type="decoder")
+        delete_HIO_general(self, type="mlp")
+
     def forward(
         self,
         hidden_states: torch.Tensor,
@@ -308,6 +539,7 @@ class LlamaDecoderLayer(nn.Module):
         use_cache: Optional[bool] = False,
         cache_position: Optional[torch.LongTensor] = None,
         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
+        Mask: Optional[torch.Tensor] = None,
         **kwargs: Unpack[FlashAttentionKwargs],
     ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
         residual = hidden_states
@@ -326,17 +558,45 @@ class LlamaDecoderLayer(nn.Module):
             position_embeddings=position_embeddings,
             **kwargs,
         )
+
+        # TAP
+        if self.tap_args.get("tap_enabled", False):
+            tap_stop_at_steps = self.tap_args.get("tap_stop_at_steps", None)
+            assert tap_stop_at_steps is not None, "tap_stop_at_steps must be provided"
+
+            if self.remain_dim_num is None and self.learnable_mask.lower() == "false":
+                self.remain_dim_num = int(self.hidden_size * self.tap_args.get("tap_remain_ratio") // 32 * 32)
+                self.HSM_mask_proxy[:, :, :self.remain_dim_num] = 0
+                self.HSM_mask_proxy = self.HSM_mask_proxy.to(hidden_states.device)
+                Mask = (
+                    1 - self.HSM_mask_proxy * torch.sigmoid(
+                        torch.tensor(self.update_step - (tap_stop_at_steps/2)).float() / (tap_stop_at_steps/2) * 7
+                    )
+                )
+            else:
+                Mask = Mask
+
+            # pruning hidden_states
+            hidden_states = (hidden_states + hidden_states @ self.HIO_A_Attn.T @ self.HIO_B_Attn.T) * Mask
+
         hidden_states = residual + hidden_states
 
         # Fully Connected
         residual = hidden_states
         hidden_states = self.post_attention_layernorm(hidden_states)
-        hidden_states = self.mlp(hidden_states)
+        hidden_states, pruning_interm_loss = self.mlp(hidden_states)
+        
+        # TAP: pruning hidden_states
+        if self.tap_args.get("tap_enabled", False):
+            hidden_states = (hidden_states + hidden_states @ self.HIO_A_MLP.T @ self.HIO_B_MLP.T) * Mask
+
         hidden_states = residual + hidden_states
 
         outputs = (hidden_states,)
         if output_attentions:
             outputs += (self_attn_weights,)
+        
+        outputs += (pruning_interm_loss,)
 
         return outputs
 
@@ -483,6 +743,34 @@ class LlamaModel(LlamaPreTrainedModel):
         # Initialize weights and apply final processing
         self.post_init()
 
+        # TAP
+        HIO_r = int(os.getenv("HIO_r", 0))
+        self.learnable_mask = os.getenv("learnable_mask", "false")
+        for layer in self.layers:
+            setattr(layer, "learnable_mask", self.learnable_mask)
+        if HIO_r > 0:
+            self.HIO_A_Emb = nn.Parameter(torch.zeros(HIO_r, config.hidden_size))
+            self.HIO_B_Emb = nn.Parameter(torch.zeros(config.hidden_size, HIO_r))
+            self.remain_dim_num = None
+            if self.learnable_mask.lower() == "true":
+                self.HSM_mask_proxy = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
+                for layer in self.layers:
+                    layer.mlp.init_HIO_for_MLP_layer(self.learnable_mask, HIO_r)
+            else:
+                self.HSM_mask_proxy = torch.ones(1, 1, config.hidden_size)
+            self.hidden_size = config.hidden_size
+
+    def merge_HIO(self, remain_ratio, learnable_mask=False):
+        merge_HIO_general(self, type="embedding", remain_ratio=remain_ratio, learnable_mask=learnable_mask, mask_proxy=self.HSM_mask_proxy)
+        for layer in self.layers:
+            layer.merge_HIO(remain_ratio, learnable_mask, mask_proxy=self.HSM_mask_proxy)
+        del self.HSM_mask_proxy
+
+    def delete_HIO(self):
+        delete_HIO_general(self, type="embedding")
+        for layer in self.layers:
+            layer.delete_HIO()
+
     def get_input_embeddings(self):
         return self.embed_tokens
 
@@ -544,6 +832,33 @@ class LlamaModel(LlamaPreTrainedModel):
 
         hidden_states = inputs_embeds
 
+        # TAP
+        if self.layers[0].tap_args.get("tap_enabled", False):
+            tap_stop_at_steps = self.layers[0].tap_args.get("tap_stop_at_steps", None)
+            assert tap_stop_at_steps is not None, "tap_stop_at_steps must be provided"
+
+            # setup static mask proxy
+            if self.remain_dim_num is None and self.learnable_mask.lower() == "false":
+                self.remain_dim_num = int(self.hidden_size * self.layers[0].tap_args.get("tap_remain_ratio") // 32 * 32)
+                self.HSM_mask_proxy[:, :, :self.remain_dim_num] = 0
+                self.HSM_mask_proxy = self.HSM_mask_proxy.to(hidden_states.device)
+            
+            # setup static / learnable mask
+            if self.learnable_mask.lower() == "true":
+                Mask = learnable_mask_dynamic_mapping_function(self.layers[0].update_step, tap_stop_at_steps, self.HSM_mask_proxy, early_stop_ratio=0.8)
+            else:
+                Mask = (
+                    1 - self.HSM_mask_proxy * torch.sigmoid(
+                        torch.tensor(self.layers[0].update_step - (tap_stop_at_steps/2)).float() / (tap_stop_at_steps/2) * 7
+                    )
+                )
+            hidden_states = (hidden_states + hidden_states @ self.HIO_A_Emb.T @ self.HIO_B_Emb.T) * Mask
+            if self.layers[0].update_step % 10 == 0:
+                try:
+                    logger.info_rank0(f"Step/Stop: {self.layers[0].update_step} / {tap_stop_at_steps}, pruned_nums: {(Mask.detach()<1e-5).sum().item()}, \nsort: {Mask.detach().sort().values},")
+                except:
+                    print((f"Step/Stop: {self.layers[0].update_step} / {tap_stop_at_steps}, pruned_nums: {(Mask.detach()<1e-5).sum().item()}, \nsort: {Mask.detach().sort().values},"))
+
         # create position embeddings to be shared across the decoder layers
         position_embeddings = self.rotary_emb(hidden_states, position_ids)
 
@@ -551,6 +866,7 @@ class LlamaModel(LlamaPreTrainedModel):
         all_hidden_states = () if output_hidden_states else None
         all_self_attns = () if output_attentions else None
 
+        pruning_interm_loss = 0
         for decoder_layer in self.layers[: self.config.num_hidden_layers]:
             if output_hidden_states:
                 all_hidden_states += (hidden_states,)
@@ -566,6 +882,7 @@ class LlamaModel(LlamaPreTrainedModel):
                     use_cache,
                     cache_position,
                     position_embeddings,
+                    Mask if (self.learnable_mask.lower() == "true") else None,
                 )
             else:
                 layer_outputs = decoder_layer(
@@ -577,13 +894,16 @@ class LlamaModel(LlamaPreTrainedModel):
                     use_cache=use_cache,
                     cache_position=cache_position,
                     position_embeddings=position_embeddings,
+                    Mask = Mask if (self.learnable_mask.lower() == "true") else None,
                     **flash_attn_kwargs,
                 )
 
             hidden_states = layer_outputs[0]
 
             if output_attentions:
+                raise NotImplementedError("Not implemented")
                 all_self_attns += (layer_outputs[1],)
+            pruning_interm_loss = pruning_interm_loss + layer_outputs[-1]
 
         hidden_states = self.norm(hidden_states)
 
@@ -596,6 +916,7 @@ class LlamaModel(LlamaPreTrainedModel):
             past_key_values=past_key_values if use_cache else None,
             hidden_states=all_hidden_states,
             attentions=all_self_attns,
+            pruning_interm_loss=pruning_interm_loss,
         )
 
     def _update_causal_mask(
@@ -830,6 +1151,8 @@ class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
             cache_position=cache_position,
             **kwargs,
         )
+        
+        pruning_interm_loss = outputs.pruning_interm_loss
 
         hidden_states = outputs.last_hidden_state
         # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
@@ -839,6 +1162,38 @@ class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
         loss = None
         if labels is not None:
             loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
+            if (self.model.learnable_mask.lower() == "true") and self.model.layers[0].update_step > 0.02*self.model.layers[0].tap_args["tap_stop_at_steps"]:
+                tap_remain_ratio = self.model.layers[0].tap_args["tap_remain_ratio"]
+                remain_dim_num = int(self.model.hidden_size * tap_remain_ratio // 32 * 32)
+                pruning_count_loss = (10 * torch.abs(binarize(self.model.HSM_mask_proxy).sum() - remain_dim_num)/remain_dim_num)
+
+                # pruning_l2_loss = 0
+                # pruning_l1_loss = 0
+
+                pruning_l2_loss = ascend_by_step(self.model.layers[0].update_step, 1.5* self.model.layers[0].tap_args["tap_stop_at_steps"], 0, 20) * (self.model.HSM_mask_proxy[self.model.HSM_mask_proxy<=0.0] + 0.5).pow(2).mean()
+                    
+                pruning_l1_loss = ascend_by_step(self.model.layers[0].update_step, 1.5* self.model.layers[0].tap_args["tap_stop_at_steps"], 0, 2)* (self.model.HSM_mask_proxy[self.model.HSM_mask_proxy<=0.0] + 0.5).abs().mean()
+
+                # if (self.model.HSM_mask_proxy>0.0).sum() > remain_dim_num:
+                #     pruning_l2_loss = 20* (self.model.HSM_mask_proxy + 0.5).pow(2).mean()
+                #     pruning_l1_loss = 2* (self.model.HSM_mask_proxy + 0.5).mean()
+                # else:
+                #     pruning_l2_loss = 20* (self.model.HSM_mask_proxy - 0.5).pow(2).mean()
+                #     pruning_l1_loss = 2* (self.model.HSM_mask_proxy - 0.5).mean()
+
+                pruning_loss = (
+                    pruning_count_loss
+                    + pruning_l2_loss
+                    + pruning_l1_loss
+                    + pruning_interm_loss
+                )
+                pure_loss = loss.item()
+                loss = loss + pruning_loss
+                if self.model.layers[0].update_step % 10 == 0:
+                    try:
+                        logger.info_rank0(f"pure_loss: {pure_loss:.3f}, pruning_loss: {pruning_loss:.3f}, gt0: {(self.model.HSM_mask_proxy>0.0).sum()}/{self.model.HSM_mask_proxy.shape[-1]}, pruning_l2_loss: {pruning_l2_loss:.3f}, pruning_l1_loss: {pruning_l1_loss:.3f}, pruning_interm_loss: {pruning_interm_loss:.3f}")
+                    except:
+                        print(f"pure_loss: {pure_loss:.3f}, pruning_loss: {pruning_loss:.3f}, gt0: {(self.model.HSM_mask_proxy>0.0).sum()}/{self.model.HSM_mask_proxy.shape[-1]}, pruning_l2_loss: {pruning_l2_loss:.3f}, pruning_l1_loss: {pruning_l1_loss:.3f}, pruning_interm_loss: {pruning_interm_loss:.3f}")
 
         return CausalLMOutputWithPast(
             loss=loss,


diff --git a/thirdparty/transformers-4.51.1/src/transformers/trainer.py b/thirdparty/transformers-4.51.1/src/transformers/trainer.py
index 727c2ca..f0752e0 100755
--- a/thirdparty/transformers-4.51.1/src/transformers/trainer.py
+++ b/thirdparty/transformers-4.51.1/src/transformers/trainer.py
@@ -180,6 +180,7 @@ from .utils import (
 from .utils.deprecation import deprecate_kwarg
 from .utils.quantization_config import QuantizationMethod
 
+from .models.llama.modeling_llama import LlamaDecoderLayer, LlamaMLP
 
 DEFAULT_CALLBACKS = [DefaultFlowCallback]
 DEFAULT_PROGRESS_CALLBACK = ProgressCallback
@@ -2510,6 +2511,16 @@ class Trainer:
                 total_updates -= 1
             for _ in range(total_updates):
                 update_step += 1
+                # TAP
+                if update_step == 19:
+                    pass
+                if hasattr(self.model, "tap_args"):
+                    tap_args = self.model.tap_args
+                    if tap_args["tap_enabled"]:
+                        for n,m in self.model.named_modules():
+                            if isinstance(m, LlamaDecoderLayer) or isinstance(m, LlamaMLP):
+                                setattr(m, "update_step", self.state.global_step)
+                                setattr(m, "tap_stop_at_steps", tap_args["tap_stop_at_steps"])
                 num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                 batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                 for i, inputs in enumerate(batch_samples):


```

</details>
