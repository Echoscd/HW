# ["DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7", "Qwen3-14B", "Qwen3-1.7B"]
model="DeepSeek-V2-Lite-Chat/" 
# ["DeepSeek-V2-Lite-Chat", "Qwen3-14B", "Qwen3-1.7B"]
served_model_name="DeepSeek-V2-Lite-Chat" 

backend=openai-chat
endpoint="/v1/chat/completions"

host=0.0.0.0
port=8066
PROJECT_DIR="/chendong"
# dataset: sharegpt
dataset_name="arxiv-summarization"
dataset_path="ccdv/arxiv-summarization"

num_prompts=50000
request_rate=1000

echo "[INFO] Dataset: ${dataset_name}"
echo "[INFO] # Prompts: ${num_prompts}"
echo "[INFO] # Request rate: ${request_rate}"

# -------------
BENCHMARK_SERVING_PY="${PROJECT_DIR}/benchmarking/benchmark_serving.py"
BENCHMARK_SAVE_DIR="${PROJECT_DIR}/benchmarking/benchmark_results_new"
export _MODEL_DIR="${PROJECT_DIR}/model_weights/${model}"
# -------------

python $BENCHMARK_SERVING_PY \
    --backend ${backend} \
    --model ${_MODEL_DIR} \
    --served-model-name ${served_model_name} \
    --endpoint ${endpoint} \
    --host ${host} \
    --port ${port} \
    --dataset-name ${dataset_name} \
    --dataset-path ${dataset_path} \
    --num-prompts ${num_prompts} \
    --request-rate ${request_rate} \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --metric-percentiles "1,25,50,75,95,99" \
    --save-result \
    --save-detailed \
    --result-dir ${BENCHMARK_SAVE_DIR}
