# predict_vllm.py

import os
import json
import argparse
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="meta-llama/Meta-Llama-3-8B", help='checkpoint of LLM')
parser.add_argument('--root_path', type=str, default="/data/UIO-LLM-SBR/mydatasets/", help='dataset root path')
parser.add_argument('--dataset', type=str, default="diginetica", help='dataset')
parser.add_argument('--idicator', type=str, default="LLM4IDRec", help='method')
parser.add_argument('--prompt_key', type=str, default="prompt", help='the key of prompts in the data file')
parser.add_argument('--target_key', type=str, default="target", help='the key of targets/labels in the data file')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--tensor_parallel_size', type=int, default=1, help='batch size')
parser.add_argument('--user_id_aug_size', type=int, default=1, help='augmentation sequence per usid')
parser.add_argument('--target_aug_ratio', type=float, default=0.1, help='Target augmentation ratio (e.g., 0.1 for 10%)')
# 注意：vLLM 自动管理 Batch，不需要手动设定 batch_size，但如果显存爆了可以限制 max_num_seqs
# parser.add_argument('--batch_size', type=int, help='batch size') 
args = parser.parse_args()

# 1. 准备数据
print("正在读取数据...")
data_path = f'{args.root_path}/{args.dataset}/train_{args.idicator}_size_{args.user_id_aug_size}_rate_{args.target_aug_ratio}.jsonl'
prompts, targets = [], []

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    ds = [json.loads(line) for line in lines]
    for d in ds:
        prompts.append(d[args.prompt_key])
        targets.append(d[args.target_key])

print(f"数据读取完成，共 {len(prompts)} 条数据。")

# 2. 准备 LoRA 路径
lora_path = f"weights/train_{args.dataset}_{args.idicator}_size_{args.user_id_aug_size}_rate_{args.target_aug_ratio}"
print(f"LoRA 路径: {lora_path}")

# 3. 初始化 vLLM 引擎
# enable_lora=True 是关键，必须开启才能挂载 LoRA
# max_lora_rank 建议设置为你训练时的 rank (如 32, 64, 128)，如果不确定可以设大一点比如 128
print("正在初始化 vLLM 引擎...")
llm = LLM(
    model=args.model_path,
    enable_lora=True, 
    max_lora_rank=128,  
    trust_remote_code=True,
    tensor_parallel_size=args.tensor_parallel_size,  # <--- 这里的数字就是你想用的显卡数量
    gpu_memory_utilization=0.9, # 显存占用率，可根据情况调整 (0.9 是默认)
    # max_model_len=2048 # 如果遇到显存不足，可以尝试取消注释这行限制长度
)

# 4. 设置采样参数 (对应你原来的 generate 参数)
# sampling_params = SamplingParams(
#     temperature=0.8,
#     top_p=0.9,
#     max_tokens=16,      # 对应 max_new_tokens
# )

sampling_params = SamplingParams(
    temperature=0,       # 【关键】设置为 0，代表贪婪解码 (Greedy)，最稳定，速度最快
    top_p=1,             # 配合 temp=0 使用
    max_tokens=12,       # 【关键】只给很少的 token，足够生成 "i12345" 即可，不需要 200
    stop=[",", "\n", " "] # 【关键】遇到逗号、换行或空格（准备生成下一个时）立即停止
)

# 5. 定义 LoRA 请求
# vLLM 需要通过 LoRARequest 对象来动态挂载 LoRA
# adapter_name 可以随便起，lora_int_id 给个 1 就行
lora_req = LoRARequest("sbr_adapter", 1, lora_path)

# 6. 开始推理 (Generate)
# vLLM 接收整个 list，内部自动进行高吞吐并行的 Batch 处理
print("开始 vLLM 加速推理...")
# prompts = prompts[:100]
outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)

# 7. 提取结果
predicted_results = []
for output in outputs:
    # output.outputs[0].text 就是生成的文本（已经去掉了 prompt）
    generated_text = output.outputs[0].text
    predicted_results.append(generated_text)

# 8. 保存结果
save_path = f'{args.root_path}/{args.dataset}/{args.idicator}_size_{args.user_id_aug_size}_rate_{args.target_aug_ratio}_predictions.json'
print(f"正在保存结果到: {save_path}")

with open(save_path, 'w', encoding='utf8') as f:
    for prompt, target, prediction in zip(prompts, targets, predicted_results):
        line = {
            'prompt': prompt,
            'target': target,
            'prediction': prediction
        }
        f.write(json.dumps(line, ensure_ascii=False) + '\n')

print("推理完成！")