#!/bin/bash

set -e

# Default configuration
PARADIGM="ethical_information_access"
QUESTIONS_FILE="data/mcqa-entries.json"
MODEL="Qwen3-1.7B Qwen3-4B Qwen3-8B"
ENGINE="vllm"
API_URL="http://localhost:8000/v1/chat/completions"
K_RUNS="5"
OUTPUT_DIR="results"
SEED="42"
TEMPERATURE="0.7"
MAX_TOKENS="32"

MODELS_ARGS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --paradigm|-p)
            PARADIGM="$2"
            shift 2
            ;;
        --questions|-q)
            QUESTIONS_FILE="$2"
            shift 2
            ;;
        --models|-m)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                MODELS_ARGS+=("$1")
                shift
            done
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --k-runs|-k)
            K_RUNS="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed|-s)
            SEED="$2"
            shift 2
            ;;
        --temperature|-t)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --list|-l)
            python run_experiment.py --list
            exit 0
            ;;
        --help|-h)
            cat << EOF
Run editorial faithfulness experiments

Usage:
    $0 [OPTIONS]

Options:
    --paradigm, -p PARADIGM          Paradigm to run (default: ethical_information_access)
    --questions, -q FILE            Questions JSON file (default: data/mcqa-entries.json)
    --models, -m MODEL [MODEL ...]  Model name(s) to evaluate (default: Qwen3-1.7B Qwen3-4B Qwen3-8B)
    --engine ENGINE                 Engine type: http or vllm (default: vllm)
    --api-url URL                   API URL for HTTP engine
    --k-runs, -k N                  Number of runs per condition (default: 5)
    --output-dir, -o DIR            Output directory (default: results)
    --seed, -s N                    Random seed (default: 42)
    --temperature, -t FLOAT         Sampling temperature (default: 0.7)
    --max-tokens N                  Max tokens to generate (default: 512)
    --list, -l                      List available paradigms and models
    --help, -h                      Show this help message

Examples:
    $0
    $0 --models Qwen3-4B Qwen3-8B
    $0 --engine http --models Qwen3-4B
    $0 --paradigm ethical_information_access --questions data/mcqa-entries.json \\
       --models Qwen3-4B Qwen3-8B --k-runs 10 --temperature 0.8

Environment Variables:
    PARADIGM, QUESTIONS_FILE, MODEL, ENGINE, API_URL, K_RUNS,
    OUTPUT_DIR, SEED, TEMPERATURE, MAX_TOKENS
EOF
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#MODELS_ARGS[@]} -gt 0 ]]; then
    MODELS_STR="${MODELS_ARGS[*]}"
else
    MODELS_STR="$MODEL"
fi

CMD=(
    python3 run_experiment.py
    --paradigm "$PARADIGM"
    --questions "$QUESTIONS_FILE"
    --models $MODELS_STR
    --engine "$ENGINE"
    --k-runs "$K_RUNS"
    --output-dir "$OUTPUT_DIR"
    --seed "$SEED"
    --temperature "$TEMPERATURE"
    --max-tokens "$MAX_TOKENS"
)

if [[ "$ENGINE" == "http" ]]; then
    CMD+=(--api-url "$API_URL")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
