# PDC-Bench

Code and data for *PDC-Bench* (Findings of EACL 2027).
Authors: Kosuke Fujita, Daiki Nishiyama, Jun Sakuma (Institute of Science Tokyo, RIKEN).

Benchmark for evaluating how faithfully LLMs keep a semantic privacy constraint on the
granularity of in-context personal-attribute disclosure. See the paper for the definitions:
the tau/kappa constraint framework, Violation / Semantic Leakage / Unnecessary Exposure,
and the coverage measure gamma.

## Contents

- `kg/` — knowledge graphs for the four domains (`kg_geo.json` 128 / `kg_age.json` 60 /
  `kg_race.json` 54 / `kg_occ.json` 81 nodes; 323 total). `occ_base_map.json` merges
  near-duplicate occupation leaves (44 -> 37); it is used for a robustness check of the
  coverage measure gamma against the choice of occupation leaf set.
- `data/dataset_{geo,age,race,occ}.json` — benchmark samples, each a triple of
  profile x task x constraint: geography 6,900, age 4,592, race and ethnicity 4,560,
  occupation 4,300, for 20,352 entries per evaluated model. Of these, **16,105 are the main
  samples reported in the paper**; the remaining 4,247 are the explicit full-allowance
  condition CF (4,019, analyzed only in the CN-vs-CF appendix) and candidate-only
  race-and-ethnicity nodes (228, excluded from all metrics). All profiles are fully
  synthetic and describe no real individual.
  Sample ids have the form `<node>_<condition>_<task>` (e.g., `SAN_FRANCISCO_C3_T1`).
- `data/excluded_outputs.json` — the 100 (model, sample) outputs used for judge
  calibration and for checking the judge against overfitting; excluded from every
  reported metric.
- `queries/` — task sets (`tasks_<domain>.json`) and constraint texts
  (`constraints_<domain>.json`) per domain. Condition ids follow the paper: CN (no
  constraint), C1 (full prohibition), C2–C4 (permitted up to a given level; the levels
  differ by domain), and CF (explicit full allowance). `tau_level` gives the permitted
  ceiling of each condition. For discrimination-sensitive tasks, `discriminatory_use`
  describes the use of the attribute that the task is designed to elicit.
- `judge/judge_prompt_reference.md` — the judge prompt, all six in-context exemplars,
  the per-domain vocabulary instructions, and the factual-synonym table, reproduced verbatim.
- `annotation/annotation_guideline_ja.md` — the written guideline given to the eight human
  annotators of the judge-validation study (Japanese original), and
  `annotation/annotation_guideline_en.md`, its English translation.
- `eval/` — code:
  - `run_inference_api.py` (OpenAI / Anthropic models) and `run_inference_vllm.py`
    (open-weight models with vLLM): inference
  - `judge.py`: judge prompt and the mapping of judge answers to KG nodes
  - `run_grounding.py`: grounding of model outputs with the judge
  - `analyze.py`: the per-model results tables and the analyses in the Results section
    (semantic leakage and string-matching misses, constraint effect, granularity gap and
    tau-kappa cells, task relevance, model families, domains)
  - `aggregate.py`: per-condition counts and the excess-coverage measure (delta gamma)
- `requirements.txt` — Python dependencies with the versions used for the paper.

## Certainty criteria

The judge reports identifiable values under three certainty criteria, named `certain`,
`inferred`, and `stereotype`. All results in the paper use the **inferred** criterion;
violation rates under `certain` alone and up to `stereotype` are reported in an appendix.

## Usage

Python 3.11. Install the API clients with `pip install -r requirements.txt`; vLLM is needed only
for open-weight inference (see `requirements.txt`). The metric scripts use only the standard
library. API keys are read from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
All results are written under `results/`.

### 1. Inference

Each model receives the role instruction and the profile as the system message, and the
constraint text followed by the task as the user message. Save the outputs as
`results/outputs/<domain>/<model>.json`:

    python eval/run_inference_api.py --dataset data/dataset_geo.json --model gpt-4o \
        --output results/outputs/geo/gpt4o.json
    python eval/run_inference_vllm.py --dataset data/dataset_geo.json \
        --model Qwen/Qwen3-14B --output results/outputs/geo/qwen3_14b.json --tp 1

Decoding settings are documented at the top of each script. The paper served open-weight
models with vLLM 0.24.0. The aggregation scripts look for
the file names used in the paper: `gpt4o`, `gpt4omini`, `gpt5`, `gpt5mini`, `sonnet4`,
`sonnet5`, `llama_8b`, `llama_70b`, `llama4_scout`, `qwen3_14b`, `qwen3_32b`.

### 2. Grounding

Each output is judged by the two judges independently (GPT-4o and Claude Sonnet 5):

    python eval/run_grounding.py --judge gpt-4o --model gpt4o --domain all
    python eval/run_grounding.py --judge claude-sonnet-5 --model gpt4o --domain all

Results go to `results/grounding/<judge>/<domain>/<model>.json`. Use `--limit 5` to try a few
samples first. Failed judge calls, and answers that lack a required key or have an invalid value
(`check_answer` in `eval/judge.py`), are stored as `{"error": ...}` and retried once at the end
of a run; re-running the same command fills in any that remain. The metric scripts exclude such
records from every denominator and print their number, so re-run grounding until none are left
before reporting rates (in the paper's runs, none were left).

### 3. Metrics

    python eval/analyze.py --judge gpt4o --robust sonnet5
    python eval/aggregate.py --judge gpt4o

`analyze.py` writes `results/main_tables_<judge>.json` (the per-model violation and
unnecessary-exposure rates of the main results tables, inferred criterion) and
`results/analysis_<judge>_<criterion>.json` (the remaining analyses). `aggregate.py` writes
`results/aggregate_<judge>.json` with per-condition counts and delta gamma; its rates use
all constrained samples as the denominator and are not the rates reported in the main tables.
The scripts also run on a subset of domains or models; rates with no samples (e.g., the
discrimination-sensitive rate when only the geography domain is present) are reported as
`null` in the JSON and `n/a` in the printout.

## Licenses

Code is released under the MIT License and the data (knowledge graphs, profiles, tasks,
constraints) under CC BY 4.0; see `LICENSE`. The knowledge graphs contain only node
names, level labels, and the containment relations we defined; how each hierarchy was
built is documented in the appendix of the paper.
