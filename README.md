# llm-ddpg

Off-policy RL fine-tuning of a chat language model on [rl-llm](https://github.com/ekazum/rl-llm): policy
`Qwen/Qwen3-1.7B`, reward model `Skywork/Skywork-Reward-V2-Qwen3-1.7B` (frozen first, trained in a later phase),
prompts from `HuggingFaceH4/ultrafeedback_binarized`. This repo holds what is specific to the project: the extract
stage for its dataset, the config files, the Snakemake workflow and the launchers. The generic stages and, later, the
trainer threads live in rl-llm.

Phase 1 is the prompt pipeline: the two UltraFeedback preference files become `prompts_train.parquet`,
`prompts_dev.parquet` and `prompts_test.parquet` with one schema, plus a stats JSON per stage. It runs on the CPUs
of the current node; the GPU phases that follow run as SLURM jobs.

| File                                             | Role                                                                                                   |
|--------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| `pipeline/Snakefile`                             | extract → filter → split_dev → verify_prompts → merge_stats; one command line per rule                 |
| `pipeline/extract_prompts.py`                    | `ExtractPrompts`: one ultrafeedback_binarized source to the prompts extract columns                    |
| `pipeline/merge_stats.py`                        | `MergeStats`: every stage's stats JSON into `pipeline_stats.json`                                      |
| `pipeline/pipeline_config.py`, `pipeline/cli.py` | `PipelineConfig` for `config/pipeline.yaml` and the shared command line                                |
| `config/pipeline.yaml`                           | data directory, raw sources, which source is train and which is test, HF cache                         |
| `config/prompts.yaml`                            | rl-llm's `PromptPipelineConfig`: tokenizer, length limit, dev size, row minimums                       |
| `scripts/download_inputs.sh`                     | one-time download of the two parquet files and the tokenizer; links `data` to the lab share            |
| `scripts/run_prompts.sh`                         | runs the prompt pipeline on local cores; `--dry-run` prints the plan, `--cores N` sets the parallelism |

## Setup

One environment on the lab share, with rl-llm installed from its clone:

```bash
conda create -p /private/shaham-lab/ekazum/envs/llm-ddpg python=3.12 -y
conda activate /private/shaham-lab/ekazum/envs/llm-ddpg
pip install -e ~/github/rl-llm snakemake
```

## Run Phase 1

```bash
bash scripts/download_inputs.sh       # once, from a node with internet access
bash scripts/run_prompts.sh --dry-run # the plan
bash scripts/run_prompts.sh           # run; --cores N for more parallelism
```

The download links `data` in the repo root to `/private/shaham-lab/ekazum/llm-ddpg` and fills `data/raw/`; the
tokenizer goes to the shared Hugging Face cache named in `config/pipeline.yaml`, which every stage reads offline.
A rerun continues from the first missing output.

## Outputs

Under `data/pipeline/`:

| Path                                                                           | Content                                                               |
|--------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `prompts/prompts_train.parquet`, `prompts_dev.parquet`, `prompts_test.parquet` | `prompt_id`, `prompt`, `chosen`, `rejected`, `prompt_tokens`, `split` |
| `raw/<source>.parquet`, `filtered/<source>.parquet`                            | the intermediate files per source                                     |
| `stats/<module>.json`, `stats/pipeline_stats.json`                             | per-stage counts and the merged summary                               |

Run logs go to `logs/` in the repo root. To run the workflow on other inputs, pass a YAML with the same keys as
`config/pipeline.yaml` through `--configfile`; the Snakefile hands that file to the project stages too.

## License

Apache-2.0
