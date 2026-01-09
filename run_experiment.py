"""
Experiment runner for editorial faithfulness paradigms.

Usage:
    # Single model (uses HTTP server)
    python run_experiment.py --paradigm ethical_information_access --questions data/mcqa-entries.json
    
    # Multi-model with programmatic vLLM (spins up/down engines automatically)
    python run_experiment.py --paradigm ethical_information_access --models Qwen3-4B Qwen3-8B --engine vllm
"""

import argparse
import json
import os
import datetime
import re
import random
from dataclasses import asdict
from typing import Optional, Any, List

from paradigms import (
    Paradigm,
    MCQAProblem,
    ExperimentalCondition,
    TrialResult,
    EthicalInformationAccessParadigm,
)
from engine import (
    InferenceEngine,
    VLLMEngine,
    HTTPEngine,
    ModelConfig,
    get_model_config,
    engine_context,
    MODEL_CONFIGS,
)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODELS = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]  # Default multi-model set
DEFAULT_ENGINE = "vllm"  # Default to vLLM for multi-model support
K_RUNS = 5  # Number of runs per condition per problem


# ============================================================================
# Answer Extraction
# ============================================================================

def extract_answer(text: Optional[str]) -> str:
    """Extract answer option (A, B, C, D) from model output."""
    if text is None:
        return "UNKNOWN"
    
    # Try structured format first
    match = re.search(r"Answer:\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Look for standalone letter
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1].upper()
    
    return "UNKNOWN"


def count_tokens_approx(text: str) -> int:
    """Approximate token count (words + punctuation)."""
    if not text:
        return 0
    return len(text.split())


# ============================================================================
# Data Loading
# ============================================================================

def load_problems(filepath: str) -> list[MCQAProblem]:
    """Load MCQA problems from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    problems = []
    for item in data:
        problems.append(MCQAProblem(
            id=str(item["id"]),
            question=item["question"],
            correct_answer=item["correct_answer"],
            correct_option_text=item.get("correct_option_text", ""),
            difficulty=item.get("difficulty", "medium")
        ))
    
    return problems


# ============================================================================
# Experiment Runner
# ============================================================================

def run_trial(
    paradigm: Paradigm,
    problem: MCQAProblem,
    condition: ExperimentalCondition,
    engine: InferenceEngine,
    temperature: float = 0.7,
    max_tokens: int = 512,
    control_answer: Optional[str] = None
) -> TrialResult:
    """Run a single trial for one condition."""
    
    prompt = condition.build_prompt(problem)
    output = engine.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    answer = extract_answer(output)
    
    # Compute core metrics
    cot_length = count_tokens_approx(output) if output else 0
    matches_target = (
        answer == condition.target_option 
        if condition.target_option else False
    )
    attribution = paradigm.detect_attribution(output or "", condition)
    
    # Compute flip relative to control
    answer_flipped = None
    if control_answer is not None and not condition.is_control:
        answer_flipped = (answer != control_answer)
    
    # Create result
    result = TrialResult(
        problem_id=problem.id,
        difficulty=problem.difficulty,
        condition_name=condition.name,
        target_option=condition.target_option,
        raw_output=output or "",
        extracted_answer=answer,
        cot_length=cot_length,
        matches_target=matches_target,
        manipulation_mentioned=attribution,
        control_answer=control_answer,
        answer_flipped=answer_flipped,
        extra_metrics={
            "correct_answer": problem.correct_answer,
            "is_correct": answer == problem.correct_answer,
        }
    )
    
    # Let paradigm add custom metrics
    custom_metrics = paradigm.compute_trial_metrics(result, condition, problem)
    result.extra_metrics.update(custom_metrics)
    
    return result


def run_experiment_single_model(
    paradigm: Paradigm,
    problems: list[MCQAProblem],
    engine: InferenceEngine,
    k_runs: int = K_RUNS,
    output_dir: str = "results",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> tuple[list[dict[str, TrialResult]], str, dict[str, Any]]:
    """
    Run experiment for a single model.
    
    Returns:
        Tuple of (results, run_dir, stats)
    """
    model_name = engine.model_name
    
    # Setup output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(output_dir, f"{paradigm.config.name}_{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Output files
    results_file = os.path.join(run_dir, "trials.jsonl")  # Raw trial data
    stats_file = os.path.join(run_dir, "statistics.json")  # Computed statistics
    report_file = os.path.join(run_dir, "report.txt")  # Human-readable report
    
    all_results: list[dict[str, TrialResult]] = []
    
    # Experiment metadata
    experiment_meta = {
        "paradigm": paradigm.config.name,
        "paradigm_description": paradigm.config.description,
        "model": model_name,
        "conditions": paradigm.config.condition_names,
        "control_condition": paradigm.config.control_condition,
        "k_runs": k_runs,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timestamp": timestamp,
        "num_problems": len(problems),
    }
    
    print(f"\n{'='*60}")
    print(f"Running Paradigm: {paradigm.config.name}")
    print(f"Model: {model_name}")
    print(f"Problems: {len(problems)}")
    print(f"Conditions: {paradigm.config.condition_names}")
    print(f"Runs per condition: {k_runs}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}\n")
    
    control_cond_name = paradigm.get_control_condition_name()
    
    with open(results_file, 'w') as f:
        for problem in problems:
            print(f"\n--- Problem: {problem.id} ---")
            print(f"Correct answer: {problem.correct_answer}")
            
            # Get all conditions for this problem
            conditions = paradigm.get_conditions(problem)
            
            for k in range(k_runs):
                run_results: dict[str, TrialResult] = {}
                
                # Run control first to get baseline answer
                control_condition = conditions[control_cond_name]
                control_result = run_trial(
                    paradigm, problem, control_condition, engine,
                    temperature=temperature, max_tokens=max_tokens
                )
                run_results[control_cond_name] = control_result
                control_answer = control_result.extracted_answer
                
                # Run all other conditions
                for cond_name in paradigm.config.condition_names:
                    if cond_name == control_cond_name:
                        continue
                    
                    condition = conditions[cond_name]
                    result = run_trial(
                        paradigm, problem, condition, engine,
                        temperature=temperature, max_tokens=max_tokens,
                        control_answer=control_answer
                    )
                    run_results[cond_name] = result
                
                # Log progress
                log_parts = [f"k={k+1}:"]
                for cond_name in paradigm.config.condition_names:
                    result = run_results[cond_name]
                    flip_str = ""
                    if result.answer_flipped is not None:
                        flip_str = " (flipped)" if result.answer_flipped else " (same)"
                    log_parts.append(f"{cond_name}={result.extracted_answer}{flip_str}")
                print("  " + ", ".join(log_parts))
                
                # Store with metadata for per-item analysis
                enriched_run_results = {
                    "problem_id": problem.id,
                    "run_k": k,
                    "results": run_results
                }
                all_results.append(enriched_run_results)
                
                # Write trial to JSONL
                trial_entry = {
                    "problem_id": problem.id,
                    "difficulty": problem.difficulty,
                    "run_k": k,
                    "model": model_name,
                    "correct_answer": problem.correct_answer,
                    "conditions": {
                        name: asdict(result) 
                        for name, result in run_results.items()
                    }
                }
                f.write(json.dumps(trial_entry) + "\n")
                f.flush()
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = paradigm.compute_statistics(all_results)
    stats["metadata"] = experiment_meta
    
    # Save statistics as JSON
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Generate human-readable report
    print("Generating report...")
    report = paradigm.generate_report(stats, report_file)
    
    print(f"\nOutput files:")
    print(f"  Trials: {results_file}")
    print(f"  Statistics: {stats_file}")
    print(f"  Report: {report_file}")
    
    return all_results, run_dir, stats


def run_experiment_multi_model(
    paradigm: Paradigm,
    problems: list[MCQAProblem],
    model_configs: List[ModelConfig],
    k_runs: int = K_RUNS,
    output_dir: str = "results",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> dict[str, list[dict[str, TrialResult]]]:
    """
    Run experiment across multiple models.
    
    Spins up vLLM engine for each model, runs experiment, then shuts down.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_dir = os.path.join(output_dir, f"{paradigm.config.name}_batch_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
    
    all_model_results = {}
    all_model_stats = {}
    
    # Batch metadata
    batch_meta = {
        "paradigm": paradigm.config.name,
        "paradigm_description": paradigm.config.description,
        "models": [c.name for c in model_configs],
        "conditions": paradigm.config.condition_names,
        "k_runs": k_runs,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timestamp": timestamp,
        "num_problems": len(problems),
    }
    
    print(f"\n{'#'*60}")
    print(f"BATCH EXPERIMENT: {paradigm.config.name}")
    print(f"Models: {[c.name for c in model_configs]}")
    print(f"Output: {batch_dir}")
    print(f"{'#'*60}\n")
    
    for i, config in enumerate(model_configs):
        print(f"\n{'='*60}")
        print(f"MODEL {i+1}/{len(model_configs)}: {config.name}")
        print(f"{'='*60}")
        
        with engine_context(config) as engine:
            results, run_dir, stats = run_experiment_single_model(
                paradigm=paradigm,
                problems=problems,
                engine=engine,
                k_runs=k_runs,
                output_dir=batch_dir,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            all_model_results[config.name] = results
            all_model_stats[config.name] = stats
    
    # Generate comparative summary (JSON + text)
    _generate_batch_summary(paradigm, all_model_results, all_model_stats, batch_meta, batch_dir)
    
    return all_model_results


def _generate_batch_summary(
    paradigm: Paradigm,
    all_results: dict[str, list[dict[str, TrialResult]]],
    all_stats: dict[str, dict[str, Any]],
    batch_meta: dict[str, Any],
    output_dir: str
):
    """Generate comparative summary across models (JSON + text)."""
    summary_json_file = os.path.join(output_dir, "batch_summary.json")
    summary_txt_file = os.path.join(output_dir, "batch_summary.txt")
    
    # Build JSON summary structure
    summary_data = {
        "metadata": batch_meta,
        "models": {},
        "comparisons": {},
    }
    
    # Extract key metrics per model
    for model_name, stats in all_stats.items():
        model_summary = {
            "total_runs": stats.get("total_runs", 0),
            "conditions": {},
        }
        
        for cond_name in paradigm.config.condition_names:
            cond_stats = stats.get("conditions", {}).get(cond_name, {})
            model_summary["conditions"][cond_name] = {
                "correct": cond_stats.get("correct", 0),
                "correct_rate": cond_stats.get("correct_rate", 0),
                "compliance_rate": cond_stats.get("compliance_rate", 0),
                "flip_rate": cond_stats.get("flip_rate", 0),
                "avg_cot_length": cond_stats.get("avg_cot_length", 0),
                "attribution": cond_stats.get("attribution", {}),
            }
        
        # Add paradigm-specific key metrics
        if "differential_compliance" in stats:
            model_summary["differential_compliance"] = stats["differential_compliance"]
        if "differential_explicit_attribution" in stats:
            model_summary["differential_explicit_attribution"] = stats["differential_explicit_attribution"]
        
        summary_data["models"][model_name] = model_summary
    
    # Build comparison tables for graphing
    for cond_name in paradigm.config.condition_names:
        summary_data["comparisons"][cond_name] = {
            model_name: {
                "correct_rate": all_stats[model_name].get("conditions", {}).get(cond_name, {}).get("correct_rate", 0),
                "compliance_rate": all_stats[model_name].get("conditions", {}).get(cond_name, {}).get("compliance_rate", 0),
                "flip_rate": all_stats[model_name].get("conditions", {}).get(cond_name, {}).get("flip_rate", 0),
            }
            for model_name in all_stats
        }
    
    # Save JSON
    with open(summary_json_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Build text summary
    lines = []
    lines.append("=" * 60)
    lines.append(f"BATCH SUMMARY: {paradigm.config.name}")
    lines.append("=" * 60)
    lines.append(f"Date: {datetime.datetime.now()}")
    lines.append(f"Models: {list(all_stats.keys())}")
    lines.append("")
    
    # Comparison table
    lines.append("--- PER-CONDITION COMPARISON ---")
    lines.append("")
    
    for cond_name in paradigm.config.condition_names:
        lines.append(f"Condition: {cond_name.upper()}")
        lines.append("-" * 40)
        lines.append(f"{'Model':<30} {'Accuracy':>10} {'Compliance':>12}")
        
        for model_name, stats in all_stats.items():
            cond_stats = stats.get("conditions", {}).get(cond_name, {})
            acc = cond_stats.get("correct_rate", 0) * 100
            comp = cond_stats.get("compliance_rate", 0) * 100
            lines.append(f"{model_name:<30} {acc:>9.1f}% {comp:>11.1f}%")
        lines.append("")
    
    # Key metrics comparison
    lines.append("--- KEY PARADIGM METRICS ---")
    lines.append("")
    
    first_model_stats = list(all_stats.values())[0] if all_stats else {}
    if "differential_compliance" in first_model_stats:
        lines.append(f"{'Model':<30} {'Diff. Compliance':>18}")
        for model_name, stats in all_stats.items():
            diff = stats.get("differential_compliance", 0) * 100
            lines.append(f"{model_name:<30} {diff:>+17.1f}%")
    
    summary_content = "\n".join(lines)
    
    with open(summary_txt_file, 'w') as f:
        f.write(summary_content)
    
    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'='*60}")
    print(summary_content)
    print(f"\nOutput files:")
    print(f"  JSON: {summary_json_file}")
    print(f"  Text: {summary_txt_file}")


# ============================================================================
# Main Entry Point
# ============================================================================

PARADIGM_REGISTRY = {
    "ethical_information_access": EthicalInformationAccessParadigm,
}


def get_paradigm(name: str) -> Paradigm:
    """Get paradigm instance by name."""
    if name not in PARADIGM_REGISTRY:
        available = list(PARADIGM_REGISTRY.keys())
        raise ValueError(f"Unknown paradigm: {name}. Available: {available}")
    
    factory = PARADIGM_REGISTRY[name]
    return factory() if callable(factory) else factory


def list_paradigms():
    """List available paradigms."""
    print("Available paradigms:")
    for name in PARADIGM_REGISTRY:
        paradigm = get_paradigm(name)
        print(f"  - {name}: {paradigm.config.description}")


def list_models():
    """List available pre-configured models."""
    print("Available pre-configured models:")
    for name, config in MODEL_CONFIGS.items():
        print(f"  - {name}: {config.model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run editorial faithfulness experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-model with vLLM (default - spins up server for each model)
  python run_experiment.py -p ethical_information_access -q data/mcqa-entries.json

  # Single model via HTTP server
  python run_experiment.py -p ethical_information_access --engine http --models Qwen3-4B

  # Custom model set
  python run_experiment.py -p ethical_information_access --models Qwen3-4B Qwen3-8B

  # List available paradigms and models
  python run_experiment.py --list
        """
    )
    
    parser.add_argument(
        "--paradigm", "-p",
        type=str,
        default="ethical_information_access",
        help="Paradigm to run"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available paradigms and models"
    )
    parser.add_argument(
        "--questions", "-q",
        type=str,
        default="data/mcqa-entries.json",
        help="Questions file to use"
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model name(s) to evaluate (default: Qwen3-1.7B Qwen3-4B Qwen3-8B)"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["http", "vllm"],
        default=DEFAULT_ENGINE,
        help="Engine type: 'http' for external server, 'vllm' for programmatic (default: vllm)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help="API URL for HTTP engine"
    )
    parser.add_argument(
        "--k-runs", "-k",
        type=int,
        default=K_RUNS,
        help="Number of runs per condition"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to generate"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_paradigms()
        print()
        list_models()
        return
    
    # Set seed
    random.seed(args.seed)
    
    # Load paradigm and problems
    paradigm = get_paradigm(args.paradigm)
    problems = load_problems(args.questions)
    
    # Run experiment(s)
    if args.engine == "http":
        # Single model via HTTP
        if len(args.models) > 1:
            print("Warning: HTTP engine only supports one model. Using first model.")
        
        engine = HTTPEngine(args.api_url, args.models[0])
        results, run_dir, stats = run_experiment_single_model(
            paradigm=paradigm,
            problems=problems,
            engine=engine,
            k_runs=args.k_runs,
            output_dir=args.output_dir,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        # Multi-model with programmatic vLLM
        model_configs = [get_model_config(name) for name in args.models]
        
        if len(model_configs) == 1:
            # Single model, still use vLLM engine
            with engine_context(model_configs[0]) as engine:
                results, run_dir, stats = run_experiment_single_model(
                    paradigm=paradigm,
                    problems=problems,
                    engine=engine,
                    k_runs=args.k_runs,
                    output_dir=args.output_dir,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
        else:
            # Multi-model batch
            run_experiment_multi_model(
                paradigm=paradigm,
                problems=problems,
                model_configs=model_configs,
                k_runs=args.k_runs,
                output_dir=args.output_dir,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()

