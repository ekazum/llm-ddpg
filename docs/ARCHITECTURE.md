# llm-ddpg Architecture

## Table of Contents

1. [Overview](#1-overview)
2. [Goals](#2-goals)
3. [Data Contract](#3-data-contract)
4. [Pipeline Overview](#4-pipeline-overview)
5. [Module Summary](#5-module-summary)
6. [Technology Stack](#6-technology-stack)
7. [Design Principles](#7-design-principles)
8. [Directory Structure](#8-directory-structure)
9. [Execution](#9-execution)
10. [Phases](#10-phases)

---

## 1. Overview

llm-ddpg is an off-policy reinforcement learning trainer for a chat language model. The policy answers a prompt with a
full response; a frozen reward model scores the prompt and the response, and a second, trained copy of the reward model
is the critic. The parts that are not specific to language models come from rl-core, ai-runtime and ai-pipeline; this
repository holds everything else: the prompt pipeline, the evaluation and training classes, the settings and the
launchers.

Phase 1 is the prompt pipeline. It goes in as raw preference parquet files, one per source, and comes out as three
prompts files, `train`, `dev` and `test`, with one schema and a stats JSON per stage. Everything runs offline on local
files; the only network access is the one-time download of the sources and the tokenizer.

---

## 2. Goals

- One prompts contract that every later reader uses: the sampling loop, the dev selection and the test evaluation.
- Prompt lengths measured the way the policy will see them, under its own tokenizer and chat template.
- A dev split that is a pure function of the keys and the settings, so a rerun or another machine produces the same
  split.
- A verify step whose stats file exists only when every check passed, so later steps can depend on it.

---

## 3. Data Contract

`PromptSchema` names the columns of a prompts parquet. The extract stage writes the first four; `FilterPrompts` adds
`prompt_tokens`; `SplitDev` adds `split` and writes one file per split, `prompts_<split>.parquet`.

| Column          | Type   | Written by | Meaning                                                                 |
|-----------------|--------|------------|-------------------------------------------------------------------------|
| `prompt_id`     | string | extract    | unique key; never in two splits                                         |
| `prompt`        | string | extract    | the user turn, plain text, no chat template                             |
| `chosen`        | string | extract    | the preferred reply of the source pair, plain text                      |
| `rejected`      | string | extract    | the other reply of the source pair, plain text                          |
| `prompt_tokens` | int    | filter     | tokens of the prompt under the chat template with the generation prompt |
| `split`         | string | split      | `train`, `dev` or `test`, equal to the file's split                     |

`PromptPipelineConfig` holds the stage settings: `tokenizer`, `chat_template_kwargs`, `max_prompt_tokens`,
`dev_size`, `hash_seed`, `min_rows`. It loads from `config/prompts.yaml` through ai-runtime's `YamlConfig`.
`PipelineConfig` holds the paths and sources of the workflow and loads from `config/pipeline.yaml`.

---

## 4. Pipeline Overview

```text
extract (per source) → filter_prompts (per source) → split_dev → verify_prompts → merge_stats
```

| Step              | Input                                       | Output                                                                                                                                     |
|-------------------|---------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `extract_prompts` | a raw source file                           | `raw/<source>.parquet` with the extract columns                                                                                            |
| `filter_prompts`  | `raw/<source>.parquet`                      | `filtered/<source>.parquet` with `prompt_tokens`                                                                                           |
| `split_dev`       | filtered train source, filtered test source | `prompts/prompts_train.parquet`, `prompts_dev.parquet`, `prompts_test.parquet`; train rows whose prompt text is in dev or test are dropped |
| `verify_prompts`  | the three prompts files                     | `stats/verify_prompts.json`, present only on pass                                                                                          |
| `merge_stats`     | `stats/verify_prompts.json`                 | `stats/pipeline_stats.json`                                                                                                                |

Every step writes its output through a temporary file and a rename, and a stats JSON named after its module, as
ai-pipeline's `Stage` does. `pipeline/Snakefile` wires the steps and runs each as its own command line. The prompt
pipeline is the only Snakemake workflow; each GPU job of the later phases has its own launcher and its own manager.

---

## 5. Module Summary

| File                                 | Class                  | Role                                                                        |
|--------------------------------------|------------------------|-----------------------------------------------------------------------------|
| `pipeline/prompt_schema.py`          | `PromptSchema`         | column names, split names, file name per split                              |
| `pipeline/prompt_pipeline_config.py` | `PromptPipelineConfig` | the stage settings, from YAML                                               |
| `pipeline/pipeline_config.py`        | `PipelineConfig`       | data directory, sources, cache location, from YAML                          |
| `pipeline/cli.py`                    | —                      | the command line shared by the extract and merge stages                     |
| `pipeline/extract_prompts.py`        | `ExtractPrompts`       | `ExtractStage`: one preference source to the extract columns                |
| `pipeline/filter_prompts.py`         | `FilterPrompts`        | `FilterStage`: text present, keys unique, templated length within the limit |
| `pipeline/split_dev.py`              | `SplitDev`             | `Stage`: seeded hash split of the train source, test passed through         |
| `pipeline/verify_prompts.py`         | `VerifyPrompts`        | `VerifyStage`: per-file and cross-file checks, stats as the pass signal     |
| `pipeline/merge_stats.py`            | `MergeStats`           | `MergeStatsStage`: every stage's stats JSON into one summary                |

Each stage module has a `main()` and runs as `python pipeline/<module>.py` from the repository root.

---

## 6. Technology Stack

| Concern                      | Choice                                                                               | Reason                                                                                      |
|------------------------------|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Language                     | Python 3.12                                                                          | the family's baseline                                                                       |
| Tables                       | pandas on parquet via pyarrow                                                        | keyed columnar files that every stage reads and writes                                      |
| Stage mechanics              | ai-pipeline `Stage`, `ExtractStage`, `FilterStage`, `VerifyStage`, `MergeStatsStage` | atomic outputs, stats JSON, verify-as-gate                                                  |
| Config                       | ai-runtime `YamlConfig` (pydantic)                                                   | typed, frozen, validated at load                                                            |
| Tokenization                 | transformers `AutoTokenizer`                                                         | the policy's own tokenizer and chat template                                                |
| Logging                      | ai-runtime `setup_logging`                                                           | one root logger to stdout, level from `LOG_LEVEL`                                           |
| Workflow                     | Snakemake for the prompt pipeline; a launcher per GPU job                            | many file-dependent CPU steps in the pipeline; one command per GPU job                      |
| Reinforcement learning parts | rl-core                                                                              | replay, checkpoint and stats threads, and the abstract classes a manager is written against |

---

## 7. Design Principles

Framework in rl-core, the rest here. A class goes to rl-core only when nothing in it is specific to language models;
every class that touches a tokenizer, a prompt or a response lives in this repository.

One contract, many readers. `PromptSchema` is the only place column names are spelled; a stage or a later reader imports
it rather than repeating strings.

Measure what the policy sees. The length filter applies the policy's chat template with its generation prompt, with the
same keyword arguments the sampling loop will use.

Deterministic data. The dev split is a hash of the key and a seed, with no random state, so the files are reproducible
and a resumed pipeline never changes them.

Verify is a gate, not a report. A verify stage raises on any error and writes its stats file only after the checks pass;
later steps depend on that file.

---

## 8. Directory Structure

```text
llm-ddpg/
  README.md
  LICENSE
  docs/
    ARCHITECTURE.md
  config/
    pipeline.yaml               PipelineConfig: data directory, sources, cache location
    prompts.yaml                PromptPipelineConfig: tokenizer, limits, dev size
  pipeline/
    Snakefile                   the prompt pipeline workflow
    cli.py
    pipeline_config.py
    prompt_schema.py
    prompt_pipeline_config.py
    extract_prompts.py
    filter_prompts.py
    split_dev.py
    verify_prompts.py
    merge_stats.py
  scripts/
    download_inputs.sh          one-time download of the sources and the tokenizer
    run_prompts.sh              runs the prompt pipeline on local cores
```

---

## 9. Execution

The prompt pipeline runs from the repository root through its launcher:

```bash
bash scripts/download_inputs.sh       # once, from a machine with network access
bash scripts/run_prompts.sh --dry-run # the plan
bash scripts/run_prompts.sh           # run; --cores N sets the parallelism
```

The workflow reruns a step only when its output is missing or older than its inputs. The tokenizer is read from the
Hugging Face cache named in `config/pipeline.yaml`; every stage runs offline against it.

---

## 10. Phases

Each phase adds parts that run on their own before the next phase starts. The parts named under rl-core are added to
that package; every other part is added here.

| Phase | Runs                                                                     | Parts added                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|-------|--------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | Prompt pipeline on CPU: train, dev and test files                        | `PromptSchema`, `PromptPipelineConfig`, extract, filter, dev split, verify, merge stats stages                                                                                                                                                                                                                                                                                                                                                           |
| 2     | Baseline job: reward check, then the untrained policy on dev and on test | `BaselineManager`, `LLMScoreTask`, `TokenPolicyModel`, `SequenceRewardModel`, `RewardModelWorker`, `RewardModelClient`, the evaluation schema and settings, `ResponseSchema`, `RunRecord`, `EvalRecordReader`; in rl-core: the abstract `EvalTask` and `ScoreTask`, `EvalConfig`, `EvalStore`                                                                                                                                                            |
| 3     | Sampling and reward loop, no learning                                    | `RLManager` skeleton, `PromptSampler`, `ActorProcessor`, `SamplingModel`, `ActorGPUWorker`, `TrajectoryBatch`, `TrajectoryCodec`, replay buffer, `Telemetry`, `StatsStore`; in rl-core: `ActorFeeder`, `RowSampler`                                                                                                                                                                                                                                      |
| 4     | Draw cycle                                                               | `ReplayReader`, `LearnerProcessor`                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 5     | Critic training, actor frozen; first checkpoints, each scored on dev     | the critic, a second `SequenceRewardModel` instance initialized from the reward model weights and trained; `LearnerGPUWorker` with the critic step, `QBatch`, `PriorityUpdater`, `CheckpointWriter`, `PolicySnapshotStore`, `LLMSelectTask` on its own thread (dev per checkpoint), `LLMTestTask` (the best checkpoint on test when training ends); in rl-core: `DDPGLearner` (the base of `LearnerGPUWorker`), the abstract `SelectTask` and `TestTask` |
| 6     | Critic replica on the second GPU                                         | second `LearnerGPUWorker`, critic gradient exchange                                                                                                                                                                                                                                                                                                                                                                                                      |
| 7     | Actor update, one replica                                                | the actor step in `LearnerGPUWorker`, KL to reference; `ActorGPUWorker` generates from the actor as it is updated                                                                                                                                                                                                                                                                                                                                        |
| 8     | Actor on both GPUs                                                       | the actor step on the second GPU, actor gradient exchange, half batches, second `ActorGPUWorker`                                                                                                                                                                                                                                                                                                                                                         |
| 9     | GRPO side baseline, own job                                              | `GRPOManager`: GRPO training on the same prompts and reward model, with the same select and test tasks                                                                                                                                                                                                                                                                                                                                                   |
