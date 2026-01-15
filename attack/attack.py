"""
Attack module for testing LLM responses to adversarial prompts.

This script sends prompts through attack templates to a target LLM
and collects responses for analysis.
"""

import argparse
import json
import os
from functools import partial

from autogen import OpenAIWrapper
from joblib import Parallel, delayed
from tqdm import tqdm

from defense.utility import load_llm_config, load_attack_template, load_harmful_prompt


def query_single(llm: OpenAIWrapper, template: str, name: str, prompt: str, max_retries: int = 3) -> dict:
    """
    Query the LLM with a single prompt.
    
    Args:
        llm: OpenAIWrapper instance
        template: Attack template with [INSERT PROMPT HERE] placeholder
        prompt: The harmful prompt to insert
        name: Identifier for this prompt
        max_retries: Number of retry attempts on failure
    
    Returns:
        Dict with 'name' and 'raw_response' keys
    """
    content = template.replace("[INSERT PROMPT HERE]", prompt)
    
    for attempt in range(max_retries):
        try:
            response = llm.create(
                messages=[{"role": "user", "content": content}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            if response.choices[0].finish_reason == "content_filter":
                return {"name": name, "raw_response": "[CONTENT_FILTERED]"}
            return {"name": name, "raw_response": response.choices[0].message.content.strip()}
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts for prompt '{name}': {e}")
                return {"name": name, "raw_response": f"[ERROR: {e}]"}
    
    return {"name": name, "raw_response": "[ERROR: Unknown]"}


def run_attack(
    model_name: str,
    template: str,
    prompts: dict,
    host: str = "127.0.0.1",
    port: int = 8000,
    n_workers: int = 8,
    cache_seed: int = 123,
    temperature: float = 1.0,
) -> list[dict]:
    """
    Run attack on LLM with parallel prompt processing.
    
    Args:
        model_name: Name of the model to attack
        template: Attack template string
        prompts: Dictionary of {name: prompt} pairs
        host: vLLM server hostname
        port: vLLM server port
        n_workers: Number of parallel workers
        cache_seed: Cache seed for reproducibility
        temperature: Sampling temperature
    
    Returns:
        List of response dictionaries
    """
    llm_config = load_llm_config(
        model_name=model_name,
        host_name=host,
        port=port,
        cache_seed=cache_seed,
        temperature=temperature,
    )
    llm = OpenAIWrapper(config_list=llm_config)
    
    # Create partial function with fixed llm and template
    query_fn = partial(query_single, llm, template)
    
    # Parallel execution at the request level
    outputs = Parallel(n_jobs=n_workers, backend="threading")(
        delayed(query_fn)(name, prompt)
        for name, prompt in tqdm(prompts.items(), desc="Attacking")
    )
    
    return outputs


def save_results(outputs: list[dict], output_path: str) -> None:
    """Save attack results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(outputs)} results to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adversarial attacks on LLMs")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Model name")
    parser.add_argument("--host", default="127.0.0.1", help="vLLM server host")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--workers", type=int, default=128, help="Number of parallel workers")
    parser.add_argument("--template", default="v1", help="Attack template name")
    parser.add_argument("--prompts", default="data/prompt/prompt_dan.json", help="Path to prompts JSON")
    parser.add_argument("--output-prefix", default="attack-dan", help="Output file prefix")
    parser.add_argument("--output-suffix", default="0", help="Output file suffix")
    parser.add_argument("--cache-seed", type=int, default=123, help="Cache seed for reproducibility")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load template and prompts
    template = load_attack_template(template_name=args.template)
    prompts = load_harmful_prompt(json_path=args.prompts)
    
    # Run attack with parallel processing
    outputs = run_attack(
        model_name=args.model,
        template=template,
        prompts=prompts,
        host=args.host,
        port=args.port,
        n_workers=args.workers,
        cache_seed=args.cache_seed,
        temperature=args.temperature,
    )
    
    # Save results
    # Sanitize model name for directory (replace / with -)
    model_dir = args.model.replace("/", "-")
    output_path = f"data/harmful_output/{model_dir}/{args.output_prefix}_{args.output_suffix}.json"
    save_results(outputs, output_path)


if __name__ == "__main__":
    main()
