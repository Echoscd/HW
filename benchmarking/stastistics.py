import json
import numpy as np


# 初始化列表存储prompt和output的长度
prompt_lengths = []
output_lengths = []
file_path = "/chendong/benchmarking/benchmark_results_new/openai-chat-1000.0qps--20251124-125317-detailed.jsonl"
# 读取JSONL文件并提取长度信息
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        if data['success']:
            prompt_lengths.append(data['prompt_length'])
            output_lengths.append(data['output_length'])

# 计算平均长度
avg_prompt_length = np.mean(prompt_lengths)
avg_output_length = np.mean(output_lengths)

# 计算分位数
percentiles = [10, 25, 50, 75, 90, 99]
prompt_percentiles = np.percentile(prompt_lengths, percentiles)
output_percentiles = np.percentile(output_lengths, percentiles)

# 打印结果
print("Prompt Statistics:")
print(f"Average Length: {avg_prompt_length:.2f}")
print("Percentiles:")
for p, length in zip(percentiles, prompt_percentiles):
    print(f"{p}th Percentile: {length:.2f}")

print("\nOutput Statistics:")
print(f"Average Length: {avg_output_length:.2f}")
print("Percentiles:")
for p, length in zip(percentiles, output_percentiles):
    print(f"{p}th Percentile: {length:.2f}")
