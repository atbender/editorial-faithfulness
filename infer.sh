CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model "Qwen/Qwen3-4B" \
    --infer_backend vllm \
    --stream true \
    --temperature 0.6 \
    --max_new_tokens 8192 \
    --top_p 0.95 \
    --top_k 20 \
    --repetition_penalty 1.1 \
    --system "Você é um assistente útil." \
    --use_hf true \