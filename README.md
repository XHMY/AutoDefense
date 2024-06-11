# AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks

[**Blog**](https://microsoft.github.io/autogen/blog/2024/03/11/AutoDefense/Defending%20LLMs%20Against%20Jailbreak%20Attacks%20with%20AutoDefense/)

## Installation

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install 'llama-cpp-python[server]'
pip install pyautogen pandas retry
```

## Prepare Inference Service Using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

Start the service on multiple GPUs (e.g., 0 and 1) with different ports (e.g., 9005 and 9006). The host name is `dgxh-1`.

```bash
export start_port=9005
export host_name=dgxh-1
for GPU in 0 1
do
    echo "Starting server on GPU $GPU on port $(($GPU + $start_port))"
    # use regular expression to replace the port number
    sed -i "s/\"port\": [0-9]\+/\"port\": $(($GPU + $start_port))/g" data/config/server_config.json
    HOST=$host_name CUDA_VISIBLE_DEVICES=$GPU python3 -m llama_cpp.server \
    --config_file data/config/server_config.json &
    sleep 5
done
```

## Response Generation

The responses are genearte by GPT-3.5-Turbo. Please fill in the token information in `data/config/llm_config_list.json` before running the following command.

```bash
python attack/attack.py
```

This command will generate the responses using combination-1 attack defined in `data/prompt/attack_prompt_template.json`.
It attacks 5 time using differt seed by default.

## Run Defense Experiments

The follow command runs the experiments of 1-Agent, 2-Agent, and 3-Agent defense. The `data/harmful_output/gpt-35-turbo-1106/attack-dan.json` represent all `json` file starting with `attack-dan` in the folder `data/harmful_output/gpt-35-turbo-1106/`. You can also pass the full path of the `json` file, and change the `num_of_repetition` in the code (default `num_of_repetition=5`). This value will be ignored if there are multiple `json` files matching the pattern.

```bash
python defense/run_defense_exp.py --model_list llama-2-13b \
--output_suffix _fp0-dan --temperature 0.7 --frequency_penalty 0.0 --presence_penalty 0.0 \
--eval_harm --chat_file data/harmful_output/gpt-35-turbo-1106/attack-dan.json \
--host_name dgxh-1 --port_start 9005 --num_of_instance 2
```

After finish the defense experiment, the output will appear in the `data/defense_output/open-llm-defense_fp0-dan/llama-2-13b` folder.

## GPT-4 Evaluation

Evaluating harmful output defense:

```bash
python evaluator/gpt4_evaluator.py \
--defense_output_dir data/defense_output/open-llm-defense_fp0-dan/llama-2-13b \
--ori_prompt_file_name prompt_dan.json
```

After finish the evaluation, the output will appear in the `data/defense_output/open-llm-defense_fp0-dan/llama-2-13b/asr.csv`.
There will be also a `score` value appears for each defense output in the output `json` file.
Please make sure you fill in the `api_key` for `gpt-4-1106-preview` in `data/config/llm_config_list.json` before running the following command. The GPT-4 evaluation is very costly. We have enabled the cache mechanism to avoid repeated evaluation.

For safe response evaluation, there is an efficient way without using GPT-4. If you know all the prompts in your dataset are regular user prompts and should not be rejected, you can use the following command to evaluate the false positive rate(FPR) of the defense output.

```bash
python evaluator/evaluate_safe.py
```

This will find all output folders in `data/defense_output` that contain the keyword `-safe` and evaluate the false positive rate(FPR).
The FPR will be saved in the `data/defense_output/defense_fp.csv` file.
