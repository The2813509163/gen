import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse
import os
from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer, LlamaMLP

parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model and save merged model.")
parser.add_argument("--base_model", type=str, required=True, help="Path or name of the base model.")
parser.add_argument("--lora_model", type=str, required=True, help="Path to the LoRA adapter.")
parser.add_argument("--output_model", type=str, required=True, help="Path to save the merged model.")
parser.add_argument("--HIO_enabled", action="store_true", help="Enable HIO.")
parser.add_argument("--HIO_r", type=int, default=None, help="HIO rank.")
parser.add_argument("--remain_ratio", type=float, required=True, help="Remain ratio.")
parser.add_argument("--learnable_mask", action="store_true", help="Use learnable mask.")
args = parser.parse_args()

if args.HIO_enabled and args.HIO_r is not None:
    os.environ["HIO_r"] = str(args.HIO_r)
if args.learnable_mask:
    os.environ["learnable_mask"] = "true"


# 1. Load the Base Model and the LoRA Adapter

# Replace with your desired base model and LoRA adapter paths/IDs
# base_model_name_or_path = "/home/kris/workspace/checkpoints/Qwen__Qwen2.5-Math-1.5B-Instruct"  # Example: Mistral 7B
# lora_model_name_or_path = "/home/kris/workspace/pruning/Qwen2.5-Math/LLaMA-Factory/saves/Qwen__Qwen2.5-Math-1.5B-Instruct/lora/sft"  #  e.g., "my_lora_adapter"
base_model_name_or_path = args.base_model
lora_model_name_or_path = args.lora_model

# Important: Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


try:
    # Load the base model (without LoRA first)
    config = AutoConfig.from_pretrained(base_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        config=config,
        torch_dtype=torch.float16, # Or torch.bfloat16 for better performance on compatible GPUs
        low_cpu_mem_usage=True,  # Reduce CPU memory usage
        device_map="auto" # Automatically place layers on GPU if available

    )
    model.eval() # Set to evaluation mode

    if args.lora_model != "skip":
        # Load the LoRA configuration
        lora_config = PeftConfig.from_pretrained(lora_model_name_or_path)
        # Load the LoRA adapter
        model = PeftModel.from_pretrained(model, lora_model_name_or_path)
        model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Add padding token to avoid issues

except Exception as e:
    print(f"Error loading model or LoRA: {e}")
    raise

# 2.  Merge LoRA into the Base Model
if args.lora_model != "skip":
    tap_args = {
        "tap_enabled": True,
        "tap_rank": 512,
        "tap_stop_at_steps": 6000,
        'tap_remain_ratio': 0.5,
        "tap_learnable_mask": True
    }
    setattr(model, "tap_args", tap_args)
    for n,m in model.named_modules():
        setattr(m, "update_step", 10000)
        setattr(m, "tap_stop_at_steps", tap_args["tap_stop_at_steps"])
        if isinstance(m, LlamaDecoderLayer) or isinstance(m, LlamaMLP):
            setattr(m, "tap_args", tap_args)

messages = [
    {"role": "system", "content": "You are a helpful assistant!"},
    {"role": "user", "content": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nPlease reason step by step, and put your final answer within \\boxed{}.\n"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    # add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

model.to("cuda")
with torch.no_grad():
    model = model.merge_and_unload()
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
raise NotImplementedError


try:
    # Disable gradients for efficient merging
    with torch.no_grad():
        model = model.merge_and_unload()
        for n, m in model.named_modules():
            if isinstance(m, LlamaModel):
                print(f"Merging HIO for {n}")
                m.merge_HIO(remain_ratio=args.remain_ratio, learnable_mask=args.learnable_mask)
                m.delete_HIO()
except Exception as e:
    print(f"Error merging LoRA: {e}")
    raise


# 3. Save the Merged Model

# output_model_path = "saves/Qwen__Qwen2.5-Math-1.5B-Instruct/lora_merged"  #  Specify where to save the merged model
output_model_path = args.output_model

try:
    # Save the merged model
    model.save_pretrained(output_model_path)

    # Optionally save the tokenizer as well
    tokenizer.save_pretrained(output_model_path)

    print(f"Merged model saved to {output_model_path}")

except Exception as e:
    print(f"Error saving merged model: {e}")
    raise


print("Successfully merged LoRA and saved the model!")