#!/usr/bin/env bash
# End-to-end run: the provided TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv (final version v14).
#
#   bash src/scripts/reproduce.sh <student_resource_dir> [first_step] [last_step]
#
# Steps (version names in brackets; README.md explains them):
#   1  base pipeline: normalise, block, prefilter, stage-1/2 LightGBM matchers, filler rule   [v6/v7]  CPU
#   2  character cross-encoder + stage-3 LightGBM stack                                       [v8]     GPU
#   3  stage-1/2 matchers retrained with the cross-encoder score as a feature                 [v9]     CPU
#   4  uncertain band: the pairs the stage-3 stack is unsure about                            [v10]    CPU
#   5  four transformer cross-encoders fine-tuned on the band                                 [v10/v11] GPU
#   6  reference stage-4 stack (three transformers + v9 score)                                [v11]    CPU
#   7  final stage-4 stack (four transformers, query competition, three seeds)                [v13]    CPU
#   8  decoder scale 1.1 on the saved stage-4 test scores                                      [v13]    CPU
#   9  countries without training labels: gated competition-aware decode -> final output     [v14]    CPU
#
# A step is skipped when its outputs already exist (delete them to rerun it). Work directories go under WORK_ROOT
# (default $HOME): er_work_v6, er_work_v8, er_work_v9, er_work_v10; outputs go to <student_resource>/output_<version>;
# logs to WORK_ROOT/er_logs. NJOBS sets the CPU workers (default: all cores but one).
# TESTW is the work folder that holds the test caches: by default the base folder of step 1 (our SageMaker runs
# kept them in er_work_v7b, a copy of the er_work_v6 test caches).
# Only for a quick test on a data subset: CE_ARGS / PT_ARGS (extra arguments of the cross-encoder scripts, e.g.
# "--device cpu --max-minutes 1"), PT_MODELS / V11_TAGS (fewer transformers), FORCE=1 / V14_ARGS (write the stage-4
# and step-9 outputs even without a significant gain). The submitted run used none of them.
set -euo pipefail
SR=$(cd "${1:?usage: reproduce.sh <student_resource_dir> [first_step] [last_step]}" && pwd)
FIRST=${2:-1}
LAST=${3:-9}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA=$SR/dataset
WORK_ROOT=${WORK_ROOT:-$HOME}
W6=$WORK_ROOT/er_work_v6
W8=$WORK_ROOT/er_work_v8
W9=$WORK_ROOT/er_work_v9
W10=$WORK_ROOT/er_work_v10
TESTW=${TESTW:-$W6}
LOGS=$WORK_ROOT/er_logs
NJOBS=${NJOBS:-$(( $(nproc) - 1 ))}
CE_ARGS=${CE_ARGS:-}
PT_ARGS=${PT_ARGS:-}
PT_MODELS=${PT_MODELS:-xlmr xlmr-large e5 xlmr-large-s2}
V11_TAGS=${V11_TAGS:-xlmr,xlmr-large,e5}
V14_ARGS=${V14_ARGS:-}
[ -n "${FORCE:-}" ] && export V13_FORCE=1
cd "$ROOT"
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1
mkdir -p "$LOGS"

step() { [ "$1" -ge "$FIRST" ] && [ "$1" -le "$LAST" ]; }
have() { local f; for f in "$@"; do [ -e "$f" ] || return 1; done; }
run() {  # run <log name> <command...>: the command's output goes to $LOGS/<log name>.log
  local name=$1; shift
  echo "$(date +%T) $name: $*"
  if ! "$@" > "$LOGS/$name.log" 2>&1; then
    tail -20 "$LOGS/$name.log" >&2
    echo "FAILED: $name (log $LOGS/$name.log)" >&2
    exit 1
  fi
}
validate() {  # validate <output dir>
  python "$SR/utils/validate_submission.py" --matching "$1/matching_results.tsv" --candidate "$1/candidate_pairs.tsv" \
    --test-dir "$DATA/test" --check-ids | tail -1
}
T0=$(date +%s)
echo "code $ROOT  data $DATA  work $WORK_ROOT  workers $NJOBS  steps $FIRST-$LAST"

# 1. [v6/v7] base pipeline (CPU): records, blocking, prefilter, both matchers out of fold, test scores, filler rule
if step 1; then
  if have "$W6/train/oof.parquet" "$W6/test/scores.parquet" "$SR/output_v7/matching_results.tsv"; then
    echo "step 1: done before, skipped"
  else
    run 1_base python -m src.run all --data-dir "$DATA" --work-dir "$W6" --out-dir "$SR/output_v7" --n-jobs "$NJOBS"
    validate "$SR/output_v7"
  fi
fi

# 2. [v8] character cross-encoder (two fold models) + stage-3 stack
if step 2; then
  have "$W6/train/oof.parquet" "$TESTW/test/scores.parquet" "$TESTW/artifacts/models.pkl" || { echo "step 2 needs step 1" >&2; exit 1; }
  if have "$W8/ce/ce_test.parquet"; then
    echo "step 2: cross-encoder scores exist, not retrained"
  else
    run 2_char_ce python src/scripts/v8_ce.py --data-dir "$DATA" --train-work "$W6" --test-work "$TESTW" --out "$W8/ce" $CE_ARGS
  fi
  if have "$W8/stack.pkl" "$W8/oof_v8.parquet" "$SR/output_v8/matching_results.tsv"; then
    echo "step 2: stage-3 stack exists, skipped"
  else
    run 2_stage3 python src/scripts/v8_stack.py --data-dir "$DATA" --train-work "$W6" --test-work "$TESTW" --ce-dir "$W8/ce" \
      --work "$W8" --out-dir "$SR/output_v8" --n-jobs "$NJOBS"
    validate "$SR/output_v8"
  fi
fi

# 3. [v9] matchers retrained with the cross-encoder score, its query rank and margin as features. The base caches
#    (records, candidates, pair features) are hard-linked, read only: the new feature file replaces the link.
if step 3; then
  have "$W8/ce/ce_train.parquet" "$W8/ce/ce_test.parquet" || { echo "step 3 needs step 2" >&2; exit 1; }
  mkdir -p "$W9/train" "$W9/test" "$W9/artifacts" "$W9/eval"
  for f in records.parquet cand_ctx.parquet cand.parquet pairs_feat.parquet pairs_feat.folds.json; do
    [ -e "$W6/train/$f" ] && { [ -e "$W9/train/$f" ] || ln "$W6/train/$f" "$W9/train/$f"; }
  done
  for f in records.parquet cand.parquet cand_ctx.parquet; do
    [ -e "$W6/test/$f" ] && { [ -e "$W9/test/$f" ] || ln "$W6/test/$f" "$W9/test/$f"; }
  done
  for f in prefilter.pkl native_stats.pkl state_merges.json; do
    [ -e "$W6/artifacts/$f" ] && { [ -e "$W9/artifacts/$f" ] || cp "$W6/artifacts/$f" "$W9/artifacts/"; }
  done
  if have "$W9/artifacts/models.pkl"; then
    echo "step 3: matchers exist, not retrained"
  else
    run 3_v9_train python -m src.run train --data-dir "$DATA" --work-dir "$W9" --n-jobs "$NJOBS" \
      --ce-train "$W8/ce/ce_train.parquet"
  fi
  if have "$W9/test/scores.parquet" "$SR/output_v9/matching_results.tsv"; then
    echo "step 3: test scores exist, skipped"
  else
    run 3_v9_predict python -m src.run predict --data-dir "$DATA" --work-dir "$W9" --out-dir "$SR/output_v9" \
      --n-jobs "$NJOBS" --ce-test "$W8/ce/ce_test.parquet"
  fi
fi

# 4. [v10] the uncertain band (stage-3 score in [0.002, 0.998]) packed for the transformers
if step 4; then
  if have "$W10/pack/meta.json" "$W10/test_p3.parquet"; then
    echo "step 4: band exists, skipped"
  else
    run 4_band python src/scripts/v10_export.py --data-dir "$DATA" --train-work "$W6" --test-work "$TESTW" \
      --v8-work "$W8" --work "$W10"
    grep -E "fold check" "$LOGS/4_band.log" | sed -E 's/^[0-9:]+ INFO //' || true
  fi
fi

# 5. [v10/v11] transformer cross-encoders, two fold models each, scored on the band (GPU)
if step 5; then
  for m in $PT_MODELS; do
    if have "$W10/pt/$m/fold0.json" "$W10/pt/$m/fold1.json"; then
      echo "step 5: $m done before, skipped"
      continue
    fi
    case $m in
      xlmr)          run 5_xlmr python src/scripts/v10_pt.py --pack "$W10/pack" --out "$W10/pt" --model xlmr $PT_ARGS ;;
      xlmr-large)    run 5_xlmr-large python src/scripts/v10_pt.py --pack "$W10/pack" --out "$W10/pt" --model xlmr-large \
                       --max-minutes 30 --no-save-model $PT_ARGS ;;
      e5)            run 5_e5 python src/scripts/v11_pt.py --pack "$W10/pack" --out "$W10/pt" --model e5 \
                       --max-minutes 35 --no-save-model $PT_ARGS ;;
      xlmr-large-s2) run 5_xlmr-large-s2 python src/scripts/v11_pt.py --pack "$W10/pack" --out "$W10/pt" --model xlmr-large \
                       --tag xlmr-large-s2 --seed 2 --swap --max-minutes 45 --no-save-model $PT_ARGS ;;
      *) echo "unknown model $m" >&2; exit 1 ;;
    esac
  done
fi

# 6. [v11] reference stage-4 stack: v8 scores + v9 stage-2 score + XLM-R base, XLM-R large, multilingual-E5
if step 6; then
  if have "$W10/stack_v11.pkl" "$W10/oof_v11.parquet" "$SR/output_v11/matching_results.tsv"; then
    echo "step 6: done before, skipped"
  else
    run 6_stage4_v11 python src/scripts/v10_stack.py --data-dir "$DATA" --train-work "$W6" --test-work "$TESTW" \
      --v8-work "$W8" --work "$W10" --out-dir "$SR/output_v11" --n-jobs "$NJOBS" --name v11 --v9-work "$W9" \
      --tags "$V11_TAGS"
    validate "$SR/output_v11"
  fi
fi

# 7. [v13] final stage 4: every complete transformer + query competition of the stage-3 / v9 scores, three seeds;
#    variants compared with v11 paired per S1; the output is written only for a gain above two standard errors
if step 7; then
  if have "$W10/test_scores_v13.parquet" "$SR/output_v13/matching_results.tsv"; then
    echo "step 7: done before, skipped"
  else
    run 7_stage4_v13 python src/scripts/v13_stack.py --data-dir "$DATA" --train-work "$W6" --test-work "$TESTW" \
      --v8-work "$W8" --work "$W10" --out-dir "$SR/output_v13" --ref v11 --france-ref "$SR/output_v11" --seeds 3 \
      --n-jobs "$NJOBS" --name v13 --v9-work "$W9"
    grep -E "BEST variant|FRANCE CHECK" "$LOGS/7_stage4_v13.log" | sed -E 's/^[0-9:]+ INFO //' || true
    have "$SR/output_v13/matching_results.tsv" || { echo "step 7 wrote no output (no significant gain)" >&2; exit 1; }
  fi
fi

# 8. [v13] decoder scale: test-like check of scales 0.9 / 1.0 / 1.1, re-decode of the saved test scores at 1.1
if step 8; then
  if have "$SR/output_v13f/matching_results.tsv"; then
    echo "step 8: done before, skipped"
  else
    run 8_decode python src/scripts/v13_decide.py --data-dir "$DATA" --train-work "$W6" --test-work "$TESTW" \
      --work "$W10" --v8-work "$W8" --write --scale 1.1 --out-dir "$SR/output_v13f" --france-ref "$SR/output_v11"
    grep -E "BEST|FRANCE CHECK" "$LOGS/8_decode.log" | sed -E 's/^[0-9:]+ INFO //' || true
  fi
fi

# 9. [v14] countries without training labels: gated competition-aware decode; labelled rows copied from step 8
if step 9; then
  if have "$SR/output_v14/matching_results.tsv"; then
    echo "step 9: done before, skipped"
  else
    run 9_unlabelled python src/scripts/v14_france.py --data-dir "$DATA" --train-work "$W6" --test-work "$TESTW" \
      --v8-work "$W8" --base "$SR/output_v13f" --out-dir "$SR/output_v14" --write $V14_ARGS
    grep -E "CHOICE|identity check|France vs base|identical to the base" "$LOGS/9_unlabelled.log" \
      | sed -E 's/^[0-9:]+ INFO //' | cut -c1-300 || true
  fi
  have "$SR/output_v14/matching_results.tsv" || { echo "step 9 wrote no output (no eligible transformation)" >&2; exit 1; }
  echo "final output: $SR/output_v14"
  validate "$SR/output_v14"
  md5sum "$SR/output_v14/matching_results.tsv" "$SR/output_v14/candidate_pairs.tsv" 2>/dev/null || true
fi
echo "DONE in $(( ($(date +%s) - T0) / 60 )) min"
