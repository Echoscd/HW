# ["DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7", "Qwen3-14B", "Qwen3-1.7B"]
model="DeepSeek-V2-Lite-Chat/" 
# ["DeepSeek-V2-Lite-Chat", "Qwen3-14B", "Qwen3-1.7B"]
served_model_name="DeepSeek-V2-Lite-Chat"

# backend=vllm.entrypoints.api_server # do not use this one because we want to expose metrics and all the extra fancy stuff
backend=vllm.entrypoints.openai.api_server

device_list="0"
device_count=1

host=0.0.0.0
port=8066

scheduling_policy="fcfs" # priority
max_num_seqs=1024 # default 1024

# ------------- DO NOT TOUCH --------------------------------

# set devices
export CUDA_VISIBLE_DEVICES=$device_list

# set model dir
export PROJECT_DIR="/chendong/"
export _MODEL_DIR="${PROJECT_DIR}/model_weights/${model}"
echo "[INFO] Model directory: ${_MODEL_DIR}"

# if do not want profile, uncomment the exports! See docs: https://docs.vllm.ai/en/v0.9.2/contributing/profiling.html?h=profile#openai-server
# export VLLM_TORCH_PROFILER_DIR="${PROJECT_DIR}/model_serving/.vllm_profile"
# echo "[WARNING] VLLM Torch profil0ing is ON! Saving to: ${VLLM_TORCH_PROFILER_DIR}"

# --------------------

# export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_USE_V1=1
python -m ${backend} \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --enable-server-load-tracking \
    --enable-force-include-usage \
    --model ${_MODEL_DIR} \
    --served-model-name ${served_model_name} \
    --tensor-parallel-size ${device_count} \
    --trust-remote-code \
    --host ${host} \
    --port ${port} \
    --scheduling-policy ${scheduling_policy} \
    --max-num-seqs ${max_num_seqs} \
    --no-enable-prefix-caching

