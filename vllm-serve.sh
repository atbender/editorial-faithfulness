export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_MODELSCOPE=False

vllm serve "Qwen/Qwen3-4B" \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 \
  --served-model-name Qwen3-4B