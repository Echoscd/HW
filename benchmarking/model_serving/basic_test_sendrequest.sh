backend=vllm.entrypoints.openai.api_server
device_list="0"
device_count=1


export host=0.0.0.0
export port=8066

prompt="Explain a complex proof in Riemannian geometry in 1 sentence."
# --------------------

# curl http://localhost:${port}/v1/models | jq
# curl http://localhost:${port}/metrics


model_name=Qwen3-14B
model_name=DeepSeek-V2-Lite-Chat


# curl http://localhost:${port}/v1/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "'"$model_name"'",
#     "prompt": "'"$prompt"'",
#     "max_tokens": 10000
#   }' | jq '.choices[0].message.content'



curl http://localhost:${port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$model_name"'",
    "messages": [
      {"role": "user", "content": "'"$prompt"'"}
    ],
    "max_completion_tokens": 10000
  }' | jq '.choices[0].message.content'

