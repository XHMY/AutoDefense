"""
Defense experiment runner for evaluating multi-agent defense strategies.

This script runs defense strategies against harmful LLM outputs and evaluates
their effectiveness at detecting jailbreak attempts.
"""

import argparse
import os
from os.path import join, exists

from defense.explicit_detector.agency.explicit_1_agent import (
    VanillaJailbreakDetector,
    VanillaJailbreakDetectorV0125,
    CoT,
    CoTV2,
    CoTV3,
)
from defense.explicit_detector.agency.explicit_2_agents import (
    AutoGenDetectorV1,
    AutoGenDetectorV0125,
)
from defense.explicit_detector.agency.explicit_3_agents import (
    AutoGenDetectorThreeAgency,
    AutoGenDetectorThreeAgencyV2,
)
from defense.explicit_detector.explicit_defense_arch import ExplicitMultiAgentDefense
from evaluator.evaluate_helper import evaluate_defense_with_response


# Registry of available defense strategies
DEFENSE_STRATEGIES = {
    "ex-1": {"defense_agency": ExplicitMultiAgentDefense, "task_agency": VanillaJailbreakDetector},
    "ex-1-0125": {"defense_agency": ExplicitMultiAgentDefense, "task_agency": VanillaJailbreakDetectorV0125},
    "ex-2": {"defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorV1},
    "ex-2-0125": {"defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorV0125},
    "ex-3": {"defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorThreeAgency},
    "ex-3-v2": {"defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorThreeAgencyV2},
    "ex-cot": {"defense_agency": ExplicitMultiAgentDefense, "task_agency": CoT},
    "ex-cot-v2": {"defense_agency": ExplicitMultiAgentDefense, "task_agency": CoTV2},
    "ex-cot-5": {"defense_agency": ExplicitMultiAgentDefense, "task_agency": CoTV3},
}

DEFAULT_STRATEGIES = ["ex-2", "ex-3", "ex-cot"]


def run_defense_evaluation(
    model: str,
    chat_file: str,
    strategies: list[str],
    output_dir: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    workers: int = 128,
    temperature: float = 0.7,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    ignore_existing: bool = True,
) -> None:
    """
    Run defense evaluation for a model against specified strategies.
    
    Args:
        model: Model name to use for defense
        chat_file: Path to harmful outputs JSON file
        strategies: List of strategy names to evaluate
        output_dir: Directory to save defense outputs
        host: vLLM server hostname
        port: vLLM server port
        workers: Number of parallel workers
        temperature: Sampling temperature
        frequency_penalty: Frequency penalty for generation
        presence_penalty: Presence penalty for generation
        ignore_existing: Skip if output file already exists
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for strategy_name in strategies:
        if strategy_name not in DEFENSE_STRATEGIES:
            print(f"Warning: Unknown strategy '{strategy_name}', skipping")
            continue
        
        strategy = DEFENSE_STRATEGIES[strategy_name]
        output_file = join(output_dir, f"{strategy_name}.json")
        
        if exists(output_file) and ignore_existing:
            print(f"Output exists, skipping: {output_file}")
            continue
        
        print(f"Running strategy '{strategy_name}' on model '{model}'")
        print(f"  Input: {chat_file}")
        print(f"  Output: {output_file}")
        
        evaluate_defense_with_response(
            task_agency=strategy["task_agency"],
            defense_agency=strategy["defense_agency"],
            chat_file=chat_file,
            defense_output_name=output_file,
            model_name=model,
            port=port,
            host_name=host,
            parallel=True,
            num_of_threads=workers,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
        )
        print(f"  Done: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run defense experiments against harmful LLM outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model configuration
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Model name")
    parser.add_argument("--host", default="127.0.0.1", help="vLLM server host")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--workers", type=int, default=128, help="Number of parallel workers")
    
    # Input/output
    parser.add_argument("--chat-file", required=True, help="Path to harmful outputs JSON file")
    parser.add_argument("--output-dir", help="Output directory (default: data/defense_output/<model>)")
    parser.add_argument("--output-suffix", default="", help="Suffix for output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    
    # Strategy selection
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        choices=list(DEFENSE_STRATEGIES.keys()),
        help="Defense strategies to evaluate",
    )
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--frequency-penalty", type=float, default=0.0, help="Frequency penalty")
    parser.add_argument("--presence-penalty", type=float, default=0.0, help="Presence penalty")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_dir = args.model.replace("/", "-")
        output_dir = f"data/defense_output/{model_dir}{args.output_suffix}"
    
    run_defense_evaluation(
        model=args.model,
        chat_file=args.chat_file,
        strategies=args.strategies,
        output_dir=output_dir,
        host=args.host,
        port=args.port,
        workers=args.workers,
        temperature=args.temperature,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        ignore_existing=not args.force,
    )


if __name__ == "__main__":
    main()
