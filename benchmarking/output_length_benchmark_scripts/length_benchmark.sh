# ["DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7", "Qwen3-14B", "Qwen3-1.7B"]
# ["DeepSeek-V2-Lite-Chat", "Qwen3-14B", "Qwen3-1.7B"]

# ===== PICK MODEL AND SERVING SETTINGS =====
device_list="0"; device_count=1; export CUDA_VISIBLE_DEVICES=$device_list
model="Qwen3-14B"; served_model_name="Qwen3-14B"; temperature=0.6
model="DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"; served_model_name="DeepSeek-V2-Lite-Chat"; temperature=0.6
backend=openai-chat; endpoint="/v1/chat/completions"; host=0.0.0.0; port=8066
export _MODEL_DIR="${PROJECT_DIR}/model_weights/${model}"
export VLLM_USE_V1=1
export PYTHONPATH="$PYTHONPATH:${PROJECT_DIR}/efficient_scheduler"

# serving parameters
scheduling_policy="fcfs"; max_num_seqs=2048; block_size=32; backend_serving=vllm.entrypoints.openai.api_server;

# datasets
dataset_name="burstgpt"; dataset_path="${PROJECT_DIR}/datasets/public_raw/BurstGPT_without_fails_2.csv"
dataset_name="sharegpt"; dataset_path="${PROJECT_DIR}/datasets/public_raw/ShareGPT_V3_unfiltered_cleaned_split.json"

# benchmark settings
num_prompts=1000
request_rate="5 20 50 100"
max_completion_tokens=2000


# --------------------------------------------------------


export BENCHMARK_SERVING_PY="${PROJECT_DIR}/benchmarking/benchmark_serving.py"
# where to save all of the the benchmark results
DATE=$(date '+%F_%H-%M-%S')
export BENCHMARK_SAVE_DIR="${PROJECT_DIR}/benchmarking/benchmark_results/temporary" # save both datasets here

echo "[INFO] Model: ${model}"
echo "[INFO] Dataset: ${dataset_name}"
echo "[INFO] Prompts: ${num_prompts}"
echo "[INFO] Request rates: ${request_rate}"
echo "[INFO] All benchmarks will be stored at ${BENCHMARK_SAVE_DIR}"


# ================== (1) Create the base dataset that stores output length ==================
BASE_DATASET_NAME="no_length_prediction.json"
echo "[INFO] (1): Creating base dataset that will store output length!"
echo "[INFO] (1): The base dataset will be stored into \n${BENCHMARK_SAVE_DIR}/${BASE_DATASET_NAME}"
python $BENCHMARK_SERVING_PY \
    --model ${_MODEL_DIR} --served-model-name ${served_model_name} \
    --backend ${backend} --endpoint ${endpoint} --host ${host} --port ${port} \
    --dataset-name ${dataset_name} --dataset-path ${dataset_path} \
    --request-rate "inf" \
    --num-prompts ${num_prompts} \
    --temperature ${temperature} \
    --set-max-completion-tokens ${max_completion_tokens} \
    --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "1,25,50,75,95,99" \
    --save-result --save-detailed \
    --result-dir ${BENCHMARK_SAVE_DIR} \
    --result-filename ${BASE_DATASET_NAME} \
    --metadata scheduler="none"
echo "[INFO] (1): Benchmark with efficient scheduler! Not passing in additional parameters."

# ================== (2) Now run benchmark again with efficient scheduler ==================
echo "[INFO] (2): Benchmark with efficient scheduler! Passing in additional length parameters!"
# BENCHMARK_DATASET_NAME="yes_length_prediction.json"
python $BENCHMARK_SERVING_PY \
    --model ${_MODEL_DIR} --served-model-name ${served_model_name} \
    --backend ${backend} --endpoint ${endpoint} --host ${host} --port ${port} \
    --dataset-name "custom_from_previous_benchmark" --dataset-path "${BENCHMARK_SAVE_DIR}/${BASE_DATASET_NAME}" \
    --request-rate ${request_rate} \
    --num-prompts ${num_prompts} \
    --temperature ${temperature} \
    --set-max-completion-tokens ${max_completion_tokens} \
    --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "1,25,50,75,95,99" \
    --save-result --save-detailed \
    --result-dir ${BENCHMARK_SAVE_DIR} \
    --pass-output-len-to-http-request \
    --pass-output-len-as-priority \
    --metadata scheduler="efficient_priority_served"
echo "[INFO] (2): Benchmark with efficient scheduler finished!"
