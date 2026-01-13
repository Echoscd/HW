# ["DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7", "Qwen3-14B", "Qwen3-1.7B"]
model="DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"
model="Qwen3-14B"
# ["DeepSeek-V2-Lite-Chat", "Qwen3-14B", "Qwen3-1.7B"]
served_model_name="DeepSeek-V2-Lite-Chat" 
served_model_name="Qwen3-14B"

# backend=vllm.entrypoints.api_server # do not use this one because we want to expose metrics and all the extra fancy stuff
backend=vllm.entrypoints.openai.api_server

device_list="0,1"
device_count=2

host=0.0.0.0
port=8066

scheduling_policy="priority" # priority
max_num_seqs=2048 # default 1024
block_size=32
# ------------- DO NOT TOUCH --------------------------------

# set devices
export CUDA_VISIBLE_DEVICES=$device_list



# set model dir
export _MODEL_DIR="${PROJECT_DIR}/model_weights/${model}"
echo "[INFO] Model directory: ${_MODEL_DIR}"

# do profile?
DO_PROFILE=false
if [ "$DO_PROFILE" = true ]; then
  export VLLM_TORCH_PROFILER_DIR="${PROJECT_DIR}/model_serving/.vllm_profile"
  echo "[WARNING] VLLM Torch profil0ing is ON! Saving to: ${VLLM_TORCH_PROFILER_DIR}"
  echo "[INFO] By default, vLLM profiling profiles both CUDA and CPU. Change gpu_worker.py -> self.profile = ... to adjust!"
fi


# --------------------

# export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_USE_V1=1

# python -m ${backend} \
#     --disable-log-requests \
#     --no-enable-prefix-caching \
#     --enable-server-load-tracking \
#     --enable-force-include-usage \o
#     --model ${_MODEL_DIR} \
#     --served-model-name ${served_model_name} \
#     --tensor-parallel-size ${device_count} \
#     --trust-remote-code \
#     --host ${host} \
#     --port ${port} \
#     --scheduling-policy ${scheduling_policy} \
#     --max-num-seqs ${max_num_seqs}

# ---------------------------

# LENGTH PREDICTION ------
# export VLLM_ENABLE_V1_MULTIPROCESSING=0

# export VLLM_SKIP_WARMUP=true
# export VLLM_ENFORCE_EAGER=1
# export VLLM_TRACE_FUNCTION=1
# python -m ${backend} \
#     --disable-log-requests \
#     --enable-server-load-tracking \
#     --enable-force-include-usage \
#     --model ${_MODEL_DIR} \
#     --served-model-name ${served_model_name} \
#     --tensor-parallel-size ${device_count} \
#     --trust-remote-code \
#     --host ${host} \
#     --port ${port} \
#     --no-enable-prefix-caching \
#     --max-num-seqs ${max_num_seqs} \
#     --gpu-memory-utilization 0.9 \
#     --scheduling-policy ${scheduling_policy}
# #     # --disable-custom-all-reduce
# #     # --enforce-eager \
# #     # --scheduler-cls "efficient_scheduler.scheduler.EfficientScheduler"
# #     # --block-size ${block_size} 2>&1 | tee output.log

# exit 0

# set efficient scheduler path and define variables
export PYTHONPATH="$PYTHONPATH:${PROJECT_DIR}/efficient_scheduler"
export VLLM_LOGGING_LEVEL=INFO
if [[ "${VLLM_LOGGING_LEVEL}" == DEBUG ]]; then
  echo "[WARNING] DEBUG MODE IS TURNED ON"
fi
export SCHEDULER_DEBUGGING=1
export KVCACHESIM_DEBUGGING=1
# export SCHEDULER_RECORD_REQUEST_STATUS_DATACOLLECT=1
# export SCHEDULER_DUMP_DIR="/home/jovyan/pvc-shared/computational_math/g84380522/llm_scheduling/efficient_scheduler/data_dump"
export SCHEDULER_USE_MEMORY_GUARD=1
export SCHEDULER_MEMORY_GUARD_PERCENTAGE=0.96
export SCHEDULER_USE_TTFT_SLA=0
python -m ${backend} \
    --disable-log-requests \
    --enable-server-load-tracking \
    --enable-force-include-usage \
    --model ${_MODEL_DIR} \
    --served-model-name ${served_model_name} \
    --tensor-para-code \
    --host ${host}llel-size ${device_count} \
    --trust-remote \
    --port ${port} \
    --no-enable-prefix-caching \
    --max-num-seqs ${max_num_seqs} \
    --gpu-memory-utilization 0.9 \
    --scheduling-policy ${scheduling_policy} \
    --scheduler-cls "efficient_scheduler.scheduler.EfficientScheduler"

exit 0

host=0.0.0.0
port=8066
prompt="Explain a complex proof in Riemannian geometry in 1 sentence."
model_name=Qwen3-14B
curl http://localhost:${port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$model_name"'",
    "messages": [
      {"role": "user", "content": "'"$prompt"'"}
    ],
    "max_completion_tokens": 400
  }' | jq '.choices[0].message.content'


    "vllm_xargs": {
        "num_output_tokens": 100
    }

# python -m ${backend} \
#     --tensor-parallel-size 2 \
#     --pipeline-parallel-size 1 \
#     --data-parallel-size 1 \
#     --distributed-executor-backend mp \
#     --max-parallel-loading-workers 2 \
#     --disable-custom-all-reduce False \
#     --disable-log-requests \
#     --no-enable-prefix-caching \
#     --enable-server-load-tracking \
#     --enable-force-include-usage \
#     --model ${_MODEL_DIR} \
#     --served-model-name ${served_model_name} \
#     --trust-remote-code \
#     --host ${host} \
#     --port ${port} \
#     --scheduling-policy ${scheduling_policy} \
#     --max-num-seqs ${max_num_seqs}

