# Business entity resolution (Team Sandipani)

This folder regenerates `matching_results.tsv` and `candidate_pairs.tsv` from the challenge's training and test files
with one command. The submitted files come from version v14 of the pipeline (public leaderboard F0.5 0.991514). The
method, the design decisions and the results are described in `Documentation_template.md` at the root of the
submission zip.

The pipeline reads only the provided dataset. The one download is the weights of three MIT-licensed pretrained
encoders from the Hugging Face hub, at the revisions pinned in `src/ptce.py`; they are fine-tuned locally on the
training pairs.

```
train/ and test/ TSVs
  -> normalise names, addresses, house numbers
  -> blocking: char 3-gram TF-IDF, top 50 S1 per S2/S3 record
  -> LightGBM prefilter (99.95% recall)                  ==> candidate_pairs.tsv (16,294,525 test pairs)
  -> stage 1 and stage 2: LightGBM matchers
  -> stage 3: character cross-encoder + LightGBM stack
  -> stage 4: four fine-tuned transformer cross-encoders on the uncertain pairs + LightGBM stack
  -> same-address filler rule, one S1 per record, expected-F0.5 subset per S1
  -> countries without training labels: gated competition-aware decode
                                                         ==> matching_results.tsv
```

## Setting up and running on a new machine

### 1. What you need

- Linux with Python 3.12 (we used AWS SageMaker Studio).
- For step 1 of the pipeline, a CPU machine with many cores and at least 128 GB of RAM (we used 48 vCPU and 184 GiB).
- For the other steps, one NVIDIA GPU with CUDA (we used an ml.g6e.16xlarge: 64 vCPU, 512 GiB RAM, one NVIDIA L40S
  with 46 GB). One GPU machine of this size can also run step 1.
- About 60 GB of free disk for the first work folder, plus room for the later ones.
- Storage: the data, the work folders and the outputs are all kept on the machine's local disk (on SageMaker
  Studio we used the home folder). The pipeline reads and writes no cloud storage, so no S3 bucket or AWS credentials
  are needed.
- Internet access on the first run, to install the packages and download the pretrained weights (about 1.1 GB for
  each base model and 2.2 GB for the large one). Nothing else is downloaded.

### 2. Put the code and the data on the machine

Unzip the submission and go to the code folder (`python -m zipfile -e` works where `unzip` is not installed):

```bash
python3 -m zipfile -e Team_Sandipani_submission.zip submission
cd submission/code/business_entity_resolution
```

The data is the challenge's `student_resource` folder. It must contain `dataset/train/` (the three source files and
`train_ground_truth.tsv`), `dataset/test/` (the three source files) and `utils/validate_submission.py`. Below,
`/path/to/student_resource` stands for its full path.

### 3. Create the Python environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last command must print `True` and the name of the GPU. `requirements.txt` pins `torch==2.14.0`; if the default
wheel does not match your CUDA driver, install the matching build of the same version from the PyTorch index (we
used 2.14.0+cu130).

#### Pretrained models (downloaded automatically)

The three pretrained models are not Python packages, so they are not in `requirements.txt` and you do not install them
yourself. The first time step 5 needs a model, the code (`src/ptce.py`) downloads it from the Hugging Face hub at the
pinned revision below and caches it in `~/.cache/huggingface`. All three are public, so no account or token is needed.
The `huggingface_hub` library that does the download is installed automatically with `transformers`.

| model | Hugging Face name | pinned revision | parameters | licence | download |
|---|---|---|---|---|---|
| XLM-RoBERTa base | `FacebookAI/xlm-roberta-base` | `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` | 279M | MIT | about 1.1 GB |
| XLM-RoBERTa large | `FacebookAI/xlm-roberta-large` | `c23d21b0620b635a76227c604d44e43a9f0ee389` | 561M | MIT | about 2.2 GB |
| multilingual-E5 base | `intfloat/multilingual-e5-base` | `d128750597153bb5987e10b1c3493a34e5a4502a` | 278M | MIT | about 1.1 GB |

XLM-RoBERTa large is used twice (two fine-tuning runs), from the same download. To download all three in advance, for
example on a machine that will have no internet access during the run, run this once from this folder:

```bash
PYTHONPATH=. python -c "from src import ptce as PT; [(PT.load_tokenizer(n, r), PT.load_model(n, r)) for n, r, _, _ in PT.MODELS.values()]"
```

### 4. Run the pipeline

```bash
nohup bash src/scripts/reproduce.sh /path/to/student_resource > ~/reproduce.log 2>&1 &
tail -f ~/reproduce.log
```

The script sets `PYTHONPATH` itself and runs the nine steps listed below in order. It prints one line when each step
starts and writes each step's full log to `~/er_logs/<step>.log`. The whole run takes about 13 hours on the hardware
above, and it ends with `DONE`.

- Work folders (intermediate files, models, scores) go to `~/er_work_v6`, `~/er_work_v8`, `~/er_work_v9` and
  `~/er_work_v10`. Set `WORK_ROOT=/other/folder` before the command to put them elsewhere.
- `NJOBS=<n>` sets the number of CPU workers (default: all cores but one; we used 63 on the GPU machine and 47 for
  step 1 on the CPU machine).
- If the run stops, run the same command again: finished steps are skipped. `bash src/scripts/reproduce.sh
  /path/to/student_resource 5 9` runs only steps 5 to 9. To redo a step, delete its outputs first.

### 5. Where the output is saved

The final files are written to

```
/path/to/student_resource/output_v14/matching_results.tsv
/path/to/student_resource/output_v14/candidate_pairs.tsv
/path/to/student_resource/output_v14/v14_france_eval.json     (the step-9 gate table)
```

The script validates them with the official validator (`--check-ids`) and prints their md5 at the end. Each earlier
version's complete output also stays in the same folder (`output_v7`, `output_v8`, `output_v9`, `output_v11`,
`output_v13`, `output_v13f`). To place the final files where the submission layout expects them:

```bash
mkdir -p output && cp /path/to/student_resource/output_v14/*.tsv output/
python3 /path/to/student_resource/utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir /path/to/student_resource/dataset/test --check-ids
```

The validator must print PASS.

## Will a rerun give the same files?

The submitted files (in `output/` of the zip) have these md5 values:

| file | md5 |
|---|---|
| `matching_results.tsv` (1,732,544 rows) | `55da6165e4bb6c9deb81328791c3a72e` |
| `candidate_pairs.tsv` (16,294,525 pairs) | `27eb5cb991976bf65d18598dfbf58015` |

- From our saved stage-4 scores, the output is identical. We re-ran steps 8 and 9 of this exact package on
  SageMaker from the saved scores of step 7, and it reproduced the v13 file (`a9bae0849410ab0fd09973d525a09608`), the
  v14 file and the candidate set byte for byte, in 11 minutes. Steps 8 and 9 contain no randomness.
- A full rerun from the raw files on the same kind of machine runs the same code with the same settings and gives a
  very close result, but not a byte-identical file, for four reasons. The transformer and character cross-encoder
  fine-tunings run for a fixed wall-clock budget per fold, so the number of training steps depends on how fast the GPU
  runs at that moment; GPU arithmetic is not bit-for-bit deterministic; LightGBM results can change slightly with the
  number of threads; and our submitted base models were trained step by step on cached features of earlier versions,
  while the script builds everything in one pass. The differences are limited to pairs close to the decision boundary.
- On a different GPU, with other library versions or with a different number of CPU workers, the differences are
  larger, mainly because the number of transformer training steps changes. `v11_pt.py` uses the pinned model revision
  and falls back to the model's latest revision only when the pinned one cannot be downloaded; each fold records the
  revision it used, its training steps and its log loss in `er_work_v10/pt/<model>/fold{0,1}.json`.

## Version names

We built the solution in versions, and each version was checked on the training data and then on the public
leaderboard. Scripts, work folders and output folders carry the number of the version that introduced them (for
example `v8_ce.py` and `er_work_v8` belong to v8). The final output is v14.

| version | what it adds | public F0.5 |
|---|---|---|
| v1 | the baseline pipeline: TF-IDF blocking, prefilter, two LightGBM matchers | |
| v2, v3 | better features (signed house-number shifts, house-number parser, composite and unit numbers, short name tokens, raw spelling), 5 folds, extra weight on distractor records | 0.987916, 0.988272 |
| v4, v5 | earlier variants of the word-role features (not submitted) | |
| v6 | label-free word-role features for the matchers | 0.988977 |
| v7 | same-address filler rule at decoding; v7 is the complete CPU base pipeline (`src/run.py`) | 0.989604 |
| v8 | character cross-encoder and the stage-3 stack | 0.990684 |
| v9 | the stage-1/2 matchers retrained with the cross-encoder score as a feature; used only as a stage-4 input | not submitted |
| v10 | transformer cross-encoders on the uncertain pairs and the stage-4 stack | 0.991434 |
| v11 | v10 plus multilingual-E5; the reference that v13 must beat | not submitted |
| v12a | cross-encoder transfer to France (rejected, not part of the code) | 0.991236 |
| v13 | final stage 4: four transformers, query-level competition, three seeds, decoder scale 1.1 | 0.991502 |
| v14 | v13 plus the gated decode for countries without training labels: the submission | 0.991514 |

## The nine steps

| step | what it runs | writes | time |
|---|---|---|---|
| 1 | `python -m src.run all`: normalise, block, prefilter, stage-1/2 matchers out of fold, test scores, filler rule (v7) | `er_work_v6`, `output_v7` | 3 h 13 min (48 vCPU) |
| 2 | `v8_ce.py` (two fold models of the character cross-encoder), `v8_stack.py` (stage 3) | `er_work_v8`, `output_v8` | 105 min (GPU) |
| 3 | `python -m src.run train/predict --ce-train/--ce-test`: matchers with the cross-encoder score (v9) | `er_work_v9`, `output_v9` | 104 min |
| 4 | `v10_export.py`: pairs whose stage-3 score is in [0.002, 0.998] | `er_work_v10/pack` | 10 min |
| 5 | `v10_pt.py`, `v11_pt.py`: XLM-RoBERTa base, XLM-RoBERTa large, multilingual-E5 base, second XLM-RoBERTa large run | `er_work_v10/pt/<model>` | 5 h 20 min (GPU) |
| 6 | `v10_stack.py --name v11`: reference stage 4 | `er_work_v10/*_v11.*`, `output_v11` | 25 min |
| 7 | `v13_stack.py`: final stage 4, written only if it beats v11 by more than two standard errors | `er_work_v10/*_v13.*`, `output_v13` | 20 min |
| 8 | `v13_decide.py --scale 1.1`: re-decodes the saved stage-4 test scores at scale 1.1 | `output_v13f` | 5 min |
| 9 | `v14_france.py`: gated decode for countries without training labels | `output_v14` (the submission) | 6 min |

Things to look for in `~/er_logs`: step 4 prints `fold check` with `1.0000` and `0` (no query record is split across
the transformer folds); steps 7 and 8 print `FRANCE CHECK` with all 259,452 unlabelled-country rows identical to
v11; step 9 prints the gate table and `CHOICE: qn@1.1`, after an identity check that re-derives all 259,452 French
rows of the step-8 output before anything is changed.

## Layout

| path | contents |
|---|---|
| `src/run.py` | base pipeline (v7): `python -m src.run {train,predict,all}` |
| `src/io_utils.py` | TSV reading (tab separated, no quoting) and output writing |
| `src/text_names.py`, `src/text_address.py`, `src/translit.py`, `src/prepare.py` | normalisation: business names (legal forms, aliases, websites, honorifics, OCR fixes), addresses (English, French and Indian abbreviations, states and regions, house, composite and unit numbers), native-script tokens; loading and folds |
| `src/blocking.py` | TF-IDF matrices, learned state-partition merges, partitioned top-K retrieval |
| `src/features.py`, `src/surface.py`, `src/roles.py` | pair and context features, raw-surface features, per-country word-role statistics |
| `src/model.py` | LightGBM out-of-fold training, stage-2 competition and consensus features, prefilter |
| `src/fillers.py`, `src/decode.py` | same-address filler rule; one-S1-per-record assignment and expected-F0.5 decoding |
| `src/charce.py`, `src/ptce.py`, `src/stack.py` | character cross-encoder, pretrained transformer cross-encoders (pinned models), stage-3/4 stack features |
| `src/evaluation.py`, `src/testlike.py`, `src/ceiling.py` | macro F0.5, paired per-S1 comparison, test-like validation, irreducible ceiling |
| `src/par.py` | multiprocessing helpers |
| `src/resources/state_map.json` | hand-written state and region name variants, keyed by country label |
| `src/scripts/reproduce.sh` | runs the nine steps in order |
| `src/scripts/v8_ce.py`, `v8_stack.py` | step 2 |
| `src/scripts/v10_export.py` | step 4 |
| `src/scripts/v10_pt.py`, `v11_pt.py` | step 5 |
| `src/scripts/v10_stack.py` | step 6 |
| `src/scripts/v13_stack.py`, `v13_decide.py` | steps 7 and 8 |
| `src/scripts/v14_france.py` | step 9 |

## Base pipeline options and validation

`python -m src.run {train,predict,all} --data-dir <SR>/dataset --work-dir <work> --out-dir <out> --n-jobs <n>` (run
from this folder). `train` normalises the training split, blocks, trains the prefilter and both matchers out of fold
and tunes the decoding rule; `predict` applies the same steps to the test split and writes both output files.
Intermediate files are cached in `<work>/<split>/` (`records`, `cand`, `cand_ctx`, `pairs_feat`, `oof` or `scores`
parquet files); `--redo records|context|features|models|decode` recomputes from that stage on and `--force`
recomputes everything. The defaults are the submitted settings: top 50 candidates with at least 0.7 of the best score,
prefilter recall 0.9995, 5 folds, token log-odds dropout 0.4, distractor weight 1.9, word roles, raw-surface features
and the filler rule on.

```bash
PYTHONPATH=. python -m src.evaluation --data-dir <SR>/dataset --work-dir ~/er_work_v6 --tag v7 --testlike
```

prints the overall and per-country macro F0.5, pair precision and recall, singleton accuracy, blocking and prefilter
recall, the ceiling and the gap to it, and with `--testlike` the score on the test-like validation (about 90% of the
unmatched training records copied, so that 40% of records match no S1, as on the test). `--compare <tag>` reports the
paired per-S1 difference against an earlier evaluation and its standard error.
