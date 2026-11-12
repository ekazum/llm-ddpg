# llm-ddpg

Off-policy RL fine-tuning of a chat language model: policy `Qwen/Qwen3-1.7B`, reward model
`Skywork/Skywork-Reward-V2-Qwen3-1.7B` (one frozen copy gives the reward; a second copy, trained, is the critic),
prompts from `HuggingFaceH4/ultrafeedback_binarized`. The parts that are not specific to language models come from
[rl-core](https://github.com/ekazum/rl-core), [ai-runtime](https://github.com/ekazum/ai-runtime) and
[ai-pipeline](https://github.com/ekazum/ai-pipeline); this repo holds everything else: the prompt pipeline, the config
files, the launchers and, phase by phase, the evaluation and training classes. `docs/ARCHITECTURE.md`
describes the design and the phases.

Phase 1 is the prompt pipeline: the two UltraFeedback preference files become `prompts_train.parquet`,
`prompts_dev.parquet` and `prompts_test.parquet` with one schema, plus a stats JSON per stage. It runs on the CPUs of
the current node; the GPU phases that follow run as SLURM jobs.

| File                                             | Role                                                                                                    |
|--------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `pipeline/Snakefile`                             | extract → filter → split_dev → verify_prompts → merge_stats; one command line per rule                  |
| `pipeline/extract_prompts.py`                    | `ExtractPrompts`: one ultrafeedback_binarized source to the prompts extract columns                     |
| `pipeline/filter_prompts.py`                     | `FilterPrompts`: drops empty, duplicate and over-long prompts; adds `prompt_tokens`                     |
| `pipeline/split_dev.py`                          | `SplitDev`: train and dev by a seeded hash of the key, test passed through, one file per split          |
| `pipeline/verify_prompts.py`                     | `VerifyPrompts`: schema, unique keys, no dev or test prompt in train; its stats JSON is the pass signal |
| `pipeline/merge_stats.py`                        | `MergeStats`: every stage's stats JSON into `pipeline_stats.json`                                       |
| `pipeline/prompt_schema.py`                      | `PromptSchema`: the columns of a prompts parquet and the file name per split                            |
| `pipeline/prompt_pipeline_config.py`             | `PromptPipelineConfig` for `config/prompts.yaml`                                                        |
| `pipeline/pipeline_config.py`, `pipeline/cli.py` | `PipelineConfig` for `config/pipeline.yaml` and the shared command line                                 |
| `config/pipeline.yaml`                           | data directory, raw sources, which source is train and which is test, HF cache                          |
| `config/prompts.yaml`                            | `PromptPipelineConfig`: tokenizer, length limit, dev size, row minimums                                 |
| `scripts/download_inputs.sh`                     | one-time download of the two parquet files and the tokenizer; links `data` to the lab share             |
| `scripts/run_prompts.sh`                         | runs the prompt pipeline on local cores; `--dry-run` prints the plan, `--cores N` sets the parallelism  |

## Setup

One environment on the lab share:

```bash
conda create -p /private/shaham-lab/ekazum/envs/llm-ddpg python=3.12 -y
conda activate /private/shaham-lab/ekazum/envs/llm-ddpg
pip install "ai-pipeline @ git+https://github.com/ekazum/ai-pipeline.git@v0.1.0" \
    transformers pandas pyarrow numpy snakemake
```

## Run Phase 1

```bash
bash scripts/download_inputs.sh       # once, from a node with internet access
bash scripts/run_prompts.sh --dry-run # the plan
bash scripts/run_prompts.sh           # run; --cores N for more parallelism
```

The download links `data` in the repo root to `/private/shaham-lab/ekazum/llm-ddpg` and fills `data/raw/`; the tokenizer
goes to the shared Hugging Face cache named in `config/pipeline.yaml`, which every stage reads offline. A rerun
continues from the first missing output.

## Run Phase 2

Phase 2 is the baseline job: the untrained policy scored by the frozen reward model on dev and on test, after a check
that the reward model prefers the `chosen` reply of each dev pair. It needs rl-core in the environment
(`pip install -e ~/github/rl-core`, which also installs torch) and one GPU.

```bash
bash scripts/download_models.sh       # once, from a node with internet access
bash scripts/submit_baseline.sh       # submits scripts/baseline_job.sh with config/eval.yaml
```

Set `limit` in `config/eval.yaml` to a few dozen for a short first run. Each run writes
`data/records/baseline_<timestamp>/` with `eval.db`, `responses_dev.parquet`, `responses_test.parquet` and
`meta.json`; the job log is `logs/baseline_<job id>.log`. `docs/ARCHITECTURE.md` section 10 lists the classes.

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
