# ["DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7", "Qwen3-14B", "Qwen3-1.7B"]
# ["DeepSeek-V2-Lite-Chat", "Qwen3-14B", "Qwen3-1.7B"]

# ===== PICK MODEL AND SERVING SETTINGS =====
device_list="0"; device_count=1; export CUDA_VISIBLE_DEVICES=$device_list
model="DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"; served_model_name="DeepSeek-V2-Lite-Chat"; temperature=0.0
model="Qwen3-14B"; served_model_name="Qwen3-14B"; temperature=0.2
backend=openai-chat; endpoint="/v1/chat/completions"; host=0.0.0.0; port=8066
export _MODEL_DIR="${PROJECT_DIR}/model_weights/${model}"
export VLLM_USE_V1=1
export PYTHONPATH="$PYTHONPATH:${PROJECT_DIR}/efficient_scheduler"


BENCHMARK_SERVING_PY="${PROJECT_DIR}/benchmarking/benchmark_serving.py"
DATE=$(date '+%F_%H-%M-%S')

# serving parameters
scheduling_policy="fcfs"; max_num_seqs=2048; block_size=32; backend_serving=vllm.entrypoints.openai.api_server;

# datasets
dataset_name="burstgpt"; dataset_path="${PROJECT_DIR}/datasets/public_raw/BurstGPT_without_fails_2.csv"
dataset_name="sharegpt"; dataset_path="${PROJECT_DIR}/datasets/public_raw/ShareGPT_V3_unfiltered_cleaned_split.json"

# benchmark settings
num_prompts=49000 # 300 #sharegpt filtered dataset len is 49000
request_rate="20 15 10 5 4 3 2.5 2 1.5 1.0 0.5"
max_completion_tokens=10000 #4500

# ------------------------------------------------------------------------------

# where to save all of the the benchmark results
BENCHMARK_SAVE_BASE_DIR="${PROJECT_DIR}/benchmarking/base" # save the base dataset here
BENCHMARK_SAVE_REAL_DIR="${PROJECT_DIR}/benchmarking/benchmark_results/length_benchmark_${DATE}" # save all real benchmarks here

# base dataset name
BASE_DATASET_NAME="baseDataset_dataset_${dataset_name}_nprompts_${num_prompts}_model_${served_model_name}_temp_${temperature}_maxcompletion_${max_completion_tokens}.json"

# -------------- CUSTOM OVERWRITING TO USE OLD BASE OR CONTINUE OLD BENCHMARK ----------------------
# should use old base???
USE_OLD_BENCHMARK_BASE="false"
if [[ "${USE_OLD_BENCHMARK_BASE}" == "true" ]]; then
    echo "[INFO] USING OLD LENGTH BASE!"
    BASE_DATASET_NAME="baseDataset_dataset_sharegpt_nprompts_1000_model_Qwen3-14B_temp_0.6_maxcompletion_10000.json"
    BASE_DATASET_NAME="baseDataset_dataset_sharegpt_nprompts_300_model_Qwen3-14B_temp_0.6_maxcompletion_4500.json"
fi

# should add to old benchmark dir?
USE_OLD_BENCHMARK_DIR="false"
if [[ "${USE_OLD_BENCHMARK_DIR}" == "true" ]]; then
    echo "[INFO] USING OLD BENCHMARK! MANUAL BENCHMARKING"
    BENCHMARK_SAVE_REAL_DIR="${PROJECT_DIR}/benchmarking/benchmark_results/length_benchmark_2025-07-22_12-06-43"
    BENCHMARK_SAVE_REAL_DIR="${PROJECT_DIR}/benchmarking/benchmark_results/length_benchmark_2025-07-23_15-44-51"
fi

mkdir -p $BENCHMARK_SAVE_BASE_DIR; mkdir -p $BENCHMARK_SAVE_REAL_DIR
# ------------------------------------------------------------------------------




echo "[INFO] Model: ${model}"
echo "[INFO] Dataset: ${dataset_name}"
echo "[INFO] Prompts: ${num_prompts}"
echo "[INFO] Request rates: ${request_rate}"
echo "[INFO] Base length benchmark  will be stored at ${BENCHMARK_SAVE_REAL_DIR}"
echo "[INFO] All benchmarks will be stored at ${BENCHMARK_SAVE_REAL_DIR}"



# if we do not use old benchmark base, then we rerun the length base
if [[ "${USE_OLD_BENCHMARK_BASE}" == "false" ]]; then

    # ================== (0) Serve with default scheduler ==================
    read -p "(0) Serve with default scheduler + sjf priority ready or not? (y/n): " yn
    [[ "$yn" =~ ^[Yy]$ ]] && echo "Proceeding..." || { echo "Exiting."; exit 1; }
    # i.e., run this now
    # : <<'END'
    # python -m ${backend_serving} \
    #     --disable-log-requests --enable-server-load-tracking --enable-force-include-usage \
    #     --model ${_MODEL_DIR} --served-model-name ${served_model_name} --trust-remote-code \
    #     --host ${host} --port ${port} \
    #     --tensor-parallel-size ${device_count} \
    #     --max-num-seqs ${max_num_seqs} \
    #     --gpu-memory-utilization 0.9 \
    #     --no-enable-prefix-caching \
    #     --scheduling-policy "priority"
    # END
    

    # ================== (1) Create the base dataset that stores output length ==================
    echo "[INFO] (1): Creating base dataset that will store output length!"
    echo "[INFO] (1): The base dataset will be stored into \n${BENCHMARK_SAVE_BASE_DIR}/${BASE_DATASET_NAME}"
    python $BENCHMARK_SERVING_PY \
        --model ${_MODEL_DIR} --served-model-name ${served_model_name} \
        --backend ${backend} --endpoint ${endpoint} --host ${host} --port ${port} \
        --dataset-name ${dataset_name} --dataset-path ${dataset_path} \
        --request-rate "inf" \
        --num-prompts ${num_prompts} \
        --temperature ${temperature} \
        --set-max-completion-tokens ${max_completion_tokens} \
        --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "1,25,50,75,95,99" \
        --save-result \
        --result-dir ${BENCHMARK_SAVE_BASE_DIR} \
        --metadata scheduler="none" \
        --max_concurrency 2000 \
        --result-filename ${BASE_DATASET_NAME}
    # should we only create the base dataset then exit? 
    ONLY_CREATE_BASE="true"
    if [[ "${ONLY_CREATE_BASE}" == "true" ]]; then
        echo "[INFO] INSTRUCTED TO ONLY CREATE THE BASE DATASET... EXITING!"
        echo "[INFO] REMINDER: Base dataset stored into \n${BENCHMARK_SAVE_BASE_DIR}/${BASE_DATASET_NAME}"
        exit 0
    fi

fi

echo "[INFO] (1): Base Benchmark to collect output length has been completed!"


# ==============================================================================================================================================
# ==============================================================================================================================================
# ============================================== BENCHMARK BEGINS ==========================================================================
# ==============================================================================================================================================
# ================== (2) Using the base dataset, benchmark with default scheduler + FCFS/SJF across the various request_rates ==================


# echo "[INFO] (2.1): Now running benchmark with default scheduler w/ FCFS!"
# python $BENCHMARK_SERVING_PY \
#     --model ${_MODEL_DIR} --served-model-name ${served_model_name} \
#     --backend ${backend} --endpoint ${endpoint} --host ${host} --port ${port} \
#     --dataset-name "custom_from_previous_benchmark" --dataset-path "${BENCHMARK_SAVE_BASE_DIR}/${BASE_DATASET_NAME}" \
#     --request-rate ${request_rate} \
#     --num-prompts ${num_prompts} \
#     --temperature ${temperature} \
#     --set-max-completion-tokens ${max_completion_tokens} \
#     --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "1,25,50,75,95,99" \
#     --save-result --save-detailed \
#     --result-dir "${BENCHMARK_SAVE_REAL_DIR}" \
#     --metadata scheduler="default_fcfs"
# echo "[INFO] (2.1): Finished!"

# echo "[INFO] (2.2): Now running benchmark with default scheduler w/ SJF!"
# python $BENCHMARK_SERVING_PY \
#     --model ${_MODEL_DIR} --served-model-name ${served_model_name} \
#     --backend ${backend} --endpoint ${endpoint} --host ${host} --port ${port} \
#     --dataset-name "custom_from_previous_benchmark" --dataset-path "${BENCHMARK_SAVE_BASE_DIR}/${BASE_DATASET_NAME}" \
#     --request-rate ${request_rate} \
#     --num-prompts ${num_prompts} \
#     --temperature ${temperature} \
#     --set-max-completion-tokens ${max_completion_tokens} \
#     --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "1,25,50,75,95,99" \
#     --save-result --save-detailed \
#     --result-dir "${BENCHMARK_SAVE_REAL_DIR}" \
#     --pass-output-len-as-priority \
#     --metadata scheduler="default_sjf"
# echo "[INFO] (2.2): Finished!"

# ================== (3) Kill the old serving, rerun with new scheduler ==================
echo "[INFO] (3): Please kill the old serve, and run with the new scheduler w/ sjf/priority!"
read -p "(3) Serve with efficient scheduler ready or not? (y/n): " yn
[[ "$yn" =~ ^[Yy]$ ]] && echo "Proceeding..." || { echo "Exiting."; exit 1; }
# i.e., run this now 
: <<'END'
python -m ${backend_serving} \
    --disable-log-requests --enable-server-load-tracking --enable-force-include-usage \
    --model ${_MODEL_DIR} --served-model-name ${served_model_name} --trust-remote-code \
    --host ${host} --port ${port} \
    --tensor-parallel-size ${device_count} \
    --max-num-seqs ${max_num_seqs} \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --scheduling-policy "priority" \
    --scheduler-cls "efficient_scheduler.scheduler.EfficientScheduler"
END


# ================== (4) Now run benchmark again with efficient scheduler ==================
# echo "[INFO] (4.1): Now running benchmark with efficient scheduler w/ FCFS"
# python $BENCHMARK_SERVING_PY \
#     --model ${_MODEL_DIR} --served-model-name ${served_model_name} \
#     --backend ${backend} --endpoint ${endpoint} --host ${host} --port ${port} \
#     --dataset-name "custom_from_previous_benchmark" --dataset-path "${BENCHMARK_SAVE_BASE_DIR}/${BASE_DATASET_NAME}" \
#     --request-rate ${request_rate} \
#     --num-prompts ${num_prompts} \
#     --temperature ${temperature} \
#     --set-max-completion-tokens ${max_completion_tokens} \
#     --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "1,25,50,75,95,99" \
#     --save-result --save-detailed \
#     --result-dir "${BENCHMARK_SAVE_REAL_DIR}" \
#     --pass-output-len-to-http-request \
#     --metadata scheduler="efficient_memgd100_fcfs"
# echo "[INFO] (4.1): Finished!"


echo "[INFO] (4.2): Now running benchmark with efficient scheduler w/ SJF!"
python $BENCHMARK_SERVING_PY \
    --model ${_MODEL_DIR} --served-model-name ${served_model_name} \
    --backend ${backend} --endpoint ${endpoint} --host ${host} --port ${port} \
    --dataset-name "custom_from_previous_benchmark" --dataset-path "${BENCHMARK_SAVE_BASE_DIR}/${BASE_DATASET_NAME}" \
    --request-rate ${request_rate} \
    --num-prompts ${num_prompts} \
    --temperature ${temperature} \
    --set-max-completion-tokens ${max_completion_tokens} \
    --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "1,25,50,75,95,99" \
    --save-result --save-detailed \
    --result-dir "${BENCHMARK_SAVE_REAL_DIR}" \
    --pass-output-len-to-http-request \
    --pass-output-len-as-priority \
    --metadata scheduler="efficient_memgd100_sjf_preempt"
echo "[INFO] (4.2): Finished!"


# ==========================================================================================
echo "Finished length benchmarking!!!"
echo "Finished length benchmarking!!!"
echo "Finished length benchmarking!!!"



# python $BENCHMARK_SERVING_PY \
#     --backend ${backend} \
#     --model ${_MODEL_DIR} \
#     --served-model-name ${served_model_name} \
#     --endpoint ${endpoint} \
#     --host ${host} \
#     --port ${port} \
#     --dataset-name ${new_dataset_name} \
#     --dataset-path ${new_dataset_path} \
#     --num-prompts ${num_prompts} \
#     --request-rate "${request_rate}" \
#     --percentile-metrics "ttft,tpot,itl,e2el" \
#     --metric-percentiles "1,25,50,75,95,99" \
#     --save-result \
#     --save-detailed \
#     --result-dir ${BENCHMARK_SAVE_DIR} \
#     --temperature ${temperature} \
#     --set-max-completion-tokens ${max_completion_tokens} \
#     --pass-output-len-to-http-request \
#     --pass-output-len-as-priority \
#     --metadata scheduler="efficient"
#     # --result-filename ${length_benchmark_filename} \

# echo "[INFO] # Benchmark WITH length has been completed!"