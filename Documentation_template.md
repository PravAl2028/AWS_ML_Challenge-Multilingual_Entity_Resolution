# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team Sandipani  
**Team Members:** Praval Anachi (team lead), Reddymalla Dilip, Sohan Dintakurthi, Pathak Akshith Kumar  
**Submission Date:** 2026-10-02

Final version: **v14**, public leaderboard F0.5 **0.991514**. `output/matching_results.tsv` is the exact file uploaded
to the portal (md5 `55da6165e4bb6c9deb81328791c3a72e`), and `output/candidate_pairs.tsv` is the candidate set that
every model stage scored (md5 `27eb5cb991976bf65d18598dfbf58015`).

How to read this document: Section 1 (about three pages) describes the whole approach: the results, the pipeline in
Figure 1, and in Section 1.1 every model we used and why. Section 2.3 lists the ten main difficulties and how we
solved them. The rest gives the technical detail and the evidence, and Appendix D explains every term and short form.

---

## 1. Executive Summary

We match every Source-2/3 record to the Source-1 business (S1) it belongs to with a fast candidate search,
gradient-boosted tree models and fine-tuned multilingual transformers, and we choose each business's matches to
maximise the expected value of the challenge metric, F0.5 (a score that counts precision twice as much as recall;
every term is explained in Appendix D). Most of our gains came from studying how the data generator creates look-alike
distractors and messy house numbers, and from checking every change on a copy of the training data that has the test's
higher share of unmatched records. France, which has no training labels, is handled with evidence that needs no labels
and that was first shown to help on the US and India.

| result | value |
|---|---|
| public leaderboard F0.5, final version (v14) | **0.991514** |
| F0.5 out of fold on the full training set (US and India) | **0.99218** |
| best F0.5 reachable from the records alone (training set) | 0.99552 |
| test candidate pairs after blocking | 16,294,525 (reduction ratio 0.99999906) |
| true matches kept by blocking and the prefilter (training) | 99.36% |
| largest model | XLM-RoBERTa large, 561M parameters, MIT licence |
| reproducibility | re-running the last two steps of the packaged code reproduced both submitted files byte for byte |

**Figure 1. The final pipeline (v14).** Pair counts are for the test set. `p2`, `p3` and `p4` are the match
probabilities given by stages 2, 3 and 4.

```
      TRAIN: 2.21M S1, 10.32M records                 TEST: 1.73M S1, 9.97M records
      US and India, with labels                       US, India and France, no labels
                  │                                                │
                  └───────────────────────────┬────────────────────┘
                                              ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ NORMALISE    names, addresses and house numbers (hand-written, generic tables)         │
  └────────────────────────────────────────────────────────────────────────────────────────┘
                                              ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ BLOCKING     character 3-gram TF-IDF, top 50 S1 per record, inside (country, state)    │
  │              test: 51.96M candidate pairs                                              │
  └────────────────────────────────────────────────────────────────────────────────────────┘
                                              ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ PREFILTER    LightGBM, keeps 99.95% of the true pairs                                  │
  │              test: 16.29M pairs  ───────────────────────────────►  candidate_pairs.tsv │
  └────────────────────────────────────────────────────────────────────────────────────────┘
                                              ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ STAGES 1, 2  two LightGBM matchers on 115 to 125 hand-built features      ──► score p2 │
  │ STAGE 3      character cross-encoder (70M parameters, trained from scratch)            │
  │              + LightGBM stack                                             ──► score p3 │
  └────────────────────────────────────────────────────────────────────────────────────────┘
                    │ US and India                                    │ France (no labels)
                    ▼                                                 │
      ┌─────────────────────────────┐                                 │
      │ uncertain pair?             │                                 │
      │ p3 between 0.002 and 0.998  │                                 │
      └─────────────────────────────┘                                 │
            │ yes: 14%        │ no: 86%,                              │
            │                 │ keep p3                               ▼
            ▼                 │                     ┌────────────────────────────────────┐
  ┌────────────────────────┐  │                     │ stage-2 score p2 only, divided     │
  │ STAGE 4                │  │                     │ by the record's total p2           │
  │ XLM-RoBERTa base  279M │  │                     │ (qn@1.1, chosen by a check         │
  │ XLM-RoBERTa large 561M │  │                     │  on the US and India data)         │
  │ XLM-RoBERTa large 561M │  │                     └────────────────────────────────────┘
  │  (2nd run, seed 2)     │  │                                       │
  │ multilingual-E5   278M │  │                                       │
  │ + v9 matcher score     │  │                                       │
  │ + LightGBM stack,      │  │                                       │
  │   3 seeds  ──► p4      │  │                                       │
  └────────────────────────┘  │                                       │
              │               │                                       │
              └───────┬───────┘                                       │
                      ▼                                               │
      ┌────────────────────────────────┐                              │
      │ score divided by the record's  │                              │
      │ total score when above 1       │                              │
      └────────────────────────────────┘                              │
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ DECODE    same-address filler rule  ──►  each record goes to its best S1  ──►          │
  │           each S1 keeps the set of records with the highest expected F0.5              │
  │           (the empty set is allowed, which protects singletons)                        │
  └────────────────────────────────────────────────────────────────────────────────────────┘
                                              ▼
                                    matching_results.tsv
```

Main ideas:

- Features that expose the generator's planted near-copies (a house number 1 to 25 higher, an added or changed legal
  form, a changed short name token, a cleaner raw spelling), and a house-number parser that reads Indian composite and
  prefixed numbers and French formats.
- A test-like validation set that reproduces the test's higher share of unmatched records, and a paired, per-S1
  significance rule for every change.
- Cross-encoders spent where they can change a decision: 14% of the training pairs carry 99% of the expected errors,
  and only those pairs are re-read by the transformers (XLM-RoBERTa base and large, multilingual-E5 base).
- Label-free statistics computed per country on each split's own candidates (word roles, same-address noise fillers,
  query-level competition), so that a country never seen in training is decided by the same kind of evidence as the
  known ones.

Section 2.3 describes the ten main difficulties we met and how we solved each one.

### 1.1 Models used and their purpose

The final solution (v14) uses eleven trained models and three rules without trained weights. Every model is trained
only on the provided training pairs. "Fold copies" means one copy per fold, so each training pair is scored by a copy
that never saw it (Section 2.2).

| # | model | what it is | why it is in the pipeline | output |
|---|---|---|---|---|
| 1 | prefilter | LightGBM (MIT); 5 fold copies | cuts the 51.96M retrieved test pairs to the ones worth scoring while keeping 99.95% of the true pairs, so the expensive models run on 16.3M pairs | `candidate_pairs.tsv` |
| 2 | stage-1 matcher | LightGBM (MIT), about 115 features; 5 fold copies | first judgement of every pair from name, address, house-number, raw-text and word-role features | probability `p1` |
| 3 | stage-2 matcher | LightGBM (MIT), about 125 features; 5 fold copies | refines `p1` by comparing a record's candidates with each other and checking agreement among an S1's confident records | probability `p2`, also the base score for countries without labels |
| 4 | character cross-encoder | 4-layer transformer trained from scratch, 70.35M parameters (PyTorch, BSD-3); 2 fold copies | reads the four raw strings (two names, two addresses) character by character and catches spelling-level differences that the hand-built features miss; it is weaker than the matchers alone, but its errors are different | score `ce` |
| 5 | stage-3 stack | LightGBM (MIT); 5 fold copies | combines `p1`, `p2` and `ce` with their query-level competition and simple record relations into one probability; it also decides which pairs are uncertain | probability `p3` |
| 6 | v9 matchers | the stage-1 and stage-2 LightGBM retrained with `ce` as a feature; 5 + 5 fold copies | a second, differently trained matcher score for stage 4 | probability `p2` (v9) |
| 7 | XLM-RoBERTa base | `FacebookAI/xlm-roberta-base` @ `e73636d4`, 279M parameters, MIT; fine-tuned, 2 fold copies | brings pretrained multilingual knowledge of names, places, spelling variants and scripts (Latin, Devanagari and other Indian scripts) to the uncertain pairs | match logit |
| 8 | XLM-RoBERTa large, run 1 | `FacebookAI/xlm-roberta-large` @ `c23d21b0`, 561M parameters, MIT; fine-tuned, 2 fold copies | the larger version of the same model, for more accurate scores on the uncertain pairs | match logit |
| 9 | XLM-RoBERTa large, run 2 | the same model fine-tuned again with seed 2, reading the S1 record first; 2 fold copies | a second large run whose errors differ from run 1, so stage 4 can average them out | match logit |
| 10 | multilingual-E5 base | `intfloat/multilingual-e5-base` @ `d1287505`, 278M parameters, MIT; fine-tuned, 2 fold copies | an encoder pretrained for text similarity, which gives stage 4 a different view of the same pairs | match logit |
| 11 | stage-4 stack | LightGBM (MIT); 3 seeds x 5 fold copies | combines the stage-3 features and score, the four transformer logits, the v9 score and the query-level competition of the strongest scores into the final probability of each uncertain US or Indian pair | probability `p4` |

| rule (no trained weights) | why it is in the pipeline |
|---|---|
| same-address filler rule | links a record whose name differs from the S1's only by a generic filler word at the S1's exact address; the filler words are found per country from the data, without labels |
| expected-F0.5 link selection | assigns each record to its best S1, then keeps for each S1 the set of records with the highest expected F0.5 (the empty set included); two settings are chosen on out-of-fold scores |
| `qn@1.1` for countries without labels | divides a record's stage-2 scores by their total, so a record split between two S1 is not linked as if it were sure; chosen by the check in Section 4 |

---

## 2. Methodology

### 2.1 Problem Analysis

We measured every property below on the raw files or on out-of-fold predictions. To decide where to work we used a
loss budget: for each type of error, the F0.5 we would gain if only that error type were fixed.

- Planted near-copies. Unmatched records that share an S1's name and street carry a house number that is higher than
  the S1's by 1 to 25 in 93% of US cases and almost never lower, while true matches have symmetric number noise (about
  0.9% in each direction). Near-copies also add a legal form (45% of US cases) or change its family (48%), which true
  matches almost never do, and they change short name tokens such as acronyms, initials, professional titles and roman
  numerals ("MTQ" to "MTW", "MD" to "SD", "II" to "ISI").
- Indian house numbers come with prefixes and composite formats ("H No 12-7-133/102", "D.No 57-9-1/1",
  "Plot No 91/B", "#13-59"). The generator shifts the last run of the first number ("40-1-140" becomes "40-1-142"), so
  a parser that keeps only the first run sees no change in 66% of Indian near-copies. A simple parser found a house
  number for only about 45% of Indian records; ours reads 91% of S1, 84% of S2 and 82% of S3 records.
- Name-only records. About 3% of the records have an empty address, and 98% of those are true matches. When their
  S1's exact name also belongs to another S1 of the same country, nothing in the record says which one it belongs to.
  These pairs (1.48% of all true pairs) cap macro F0.5 at **0.99552** on the training set even if everything else were
  perfect, and we leave them unlinked. We checked whether the owner can be guessed from how many other records each
  same-name S1 has: it cannot (44.7% correct, below chance for two candidates).
- Near-copies are cleaner than true matches. Apart from the changed detail, near-copies copy the S1's raw spelling
  (case, accents, spacing, punctuation) more faithfully than true matches do: 69% against 47% of US records equal the
  S1 name up to the legal form, and 13% against 24% are written in upper case. Features computed on cleaned text cannot
  see this, so we added features on the raw text.
- Test distractors come in clusters. On the test set, 73% of near-copy records have another record at the same
  address (13% in training), because the generator plants whole distractor businesses with several records. Features
  that reward several records agreeing with each other therefore mislead on the test, and we do not use them.
- The test has more distractors. It has about 5.75 records per S1 against 4.68 in training, so about 40% of test
  records match no S1 against 26% in training. An unsupervised estimate on the test scores gives a match rate among
  candidate pairs of 0.364 against 0.437 in training. A model and a decision rule tuned on training scores alone would
  link too much on the test.
- France appears only in the test set (259,452 S1, 15% of test S1): three regions, very generic names, French
  abbreviations and noise such as bracketed numbers ("(12) R SAINT-SIMEON"), department names in place of regions and
  "N°" prefixes. About 35% of French records carry no recognisable region. Before we fixed the address parser,
  "Saint-X" city names were read as the street, which dropped the real street's house number.
- Postcodes are practically absent in every country, and a five-digit house number must not be read as one.
- Word roles differ by country. At the S1's own address the generator changes a name in two ways. True matches
  replace or add a noise filler (US and Indian training data: `center`, `services`, `service`, `partners`; 8,269 of the
  US same-address one-word swaps). "Different business at the same address" distractors swap a vocabulary word (Indian
  training data: 5,106 such swaps, only 5.4% of them true). France has its own fillers (`fils`, `associes`,
  `services`, `groupe`, `developpement`, `france`) and its own vocabulary (`club`, `ecole`, `comite`). Word statistics
  learned from labels cannot know French words: an early model rejected French filler swaps (probability about 0.3)
  and linked about a quarter of French vocabulary swaps. On the test set that model's own expected F0.5 was 0.9945 for
  the US and 0.9927 for India but 0.9824 for France.
- The legal form separates fillers from vocabulary. At the S1's exact address, one-word swaps whose new word is a
  noise filler keep the legal form 98 to 100% of the time in every country, French fillers included (0.1 to 1.4%
  changes). Vocabulary swaps change it in 14 to 30% of pairs (training distractor swaps 24%; French `club`, `ecole`,
  `comite` 14 to 22%), as a separate business with its own legal form would.
- Where the error is. With the test's distractor load reproduced, wrong links are the largest error that can still be
  fixed (45% of the gap to the ceiling), ahead of rejected true matches (26%), records given to the wrong S1 (17%) and
  true matches lost in blocking (8%). Because F0.5 counts precision twice, wrong links became the main target.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Blocking, a two-stage LightGBM matcher, a character cross-encoder and four pretrained
transformer cross-encoders combined by two further LightGBM stacks, then a one-S1-per-record assignment and
expected-F0.5 set selection.  
**Core Innovation:** features built around how the data generator plants near-copies, and a house-number parser for
every format we met; a validation protocol that reproduces the test's distractor load; transformer cross-encoders
applied only to the uncertain pairs; and label-free, per-country evidence (word roles, a legal-form filler rule, a
competition-aware decode) that carries the model to a country it never saw, with every such rule checked on the
labelled countries first.

Figure 1 (Section 1), stage by stage:

| stage | what it does | code |
|---|---|---|
| normalisation | names (legal forms, aliases, websites, honorifics, OCR fixes), addresses (EN/FR/IN abbreviations, states and regions, house, composite and unit numbers), native-script tokens | `src/text_names.py`, `src/text_address.py`, `src/translit.py`, `src/prepare.py` |
| blocking | TF-IDF top-50 search per record inside (country, state) partitions | `src/blocking.py` |
| prefilter | LightGBM that keeps 99.95% of true pairs, giving `candidate_pairs.tsv` | `src/model.py` |
| stages 1 and 2 | LightGBM matchers: pair features, then competition and S1-consensus features | `src/features.py`, `src/roles.py`, `src/surface.py`, `src/model.py` |
| stage 3 | character cross-encoder trained from scratch + LightGBM stack | `src/charce.py`, `src/stack.py`, `src/scripts/v8_*.py` |
| v9 | stage-1/2 matchers retrained with the cross-encoder score as a feature | `src/run.py` with `--ce-train` / `--ce-test` |
| stage 4 | four transformer cross-encoders on the uncertain pairs + LightGBM stack with query-level competition, three seeds | `src/ptce.py`, `src/scripts/v10_*.py`, `v11_pt.py`, `v13_stack.py` |
| decoding | same-address filler rule, one-S1-per-record assignment, best expected-F0.5 set per S1 | `src/fillers.py`, `src/decode.py`, `src/scripts/v13_decide.py` |
| unlabelled countries | competition-aware decode of the stage-2 score, applied only if it passes a check on the labelled countries | `src/scripts/v14_france.py` |

**How the solution evolved.** We built the solution in versions. Each was checked on the training data and then on
the public leaderboard, and the code, work folders and outputs carry the number of the version that introduced them
(for example `v8_ce.py` belongs to v8). Figure 2 shows the path to the final submission and the experiments that
branched off it; Section 5 gives the training scores of every version.

**Figure 2. From the baseline to the final submission (v14).**

```
  version            what it added                                          public F0.5
  ─────────────────  ─────────────────────────────────────────────────────  ───────────
  v1   ●             baseline: TF-IDF blocking + two LightGBM matchers
       │
  v2   ●             near-copy features, house-number parser, 5 folds          0.987916
       │
  v3   ●             raw-text features, extra weight on distractors            0.988272
       │
  v6   ●             label-free word roles (for France)                        0.988977
       │
  v7   ●             same-address filler rule (for France)                     0.989604
       │
       │             ··· CPU only up to here; one GPU from v8 on ···
       │
  v8   ●             character cross-encoder + stage-3 stack                   0.990684
       │
       ├──── v9      matchers + cross-encoder score (input of stage 4)         not sent
       │
  v10  ●             XLM-R base and large on the uncertain pairs, stage 4      0.991434
       │
  v11  ●             + multilingual-E5 (the reference v13 had to beat)         not sent
       │
       ├──── v12a    cross-encoder moved to France: REJECTED                   0.991236
       │
  v13  ●             2nd XLM-R large, query competition, 3 seeds               0.991502
       │
  v14  ●             checked France decode (qn@1.1): FINAL                     0.991514
       │
       └──── v15     name-only rule for France: failed its check               not sent

  ●     main line: each version builds on the one above it
  ├──── side experiment: tried on its own branch
```

**Validation protocol.** All numbers in this document come from the following protocol. The public leaderboard
confirmed or rejected whole versions; no parameter was tuned on it.

- Five folds split by S1 (a hash of its id). Every stage is trained out of fold. The transformer folds keep each
  record, all its candidate pairs and its true-match cluster in one fold, so a fold model never scores a record it was
  trained on.
- Every change is compared with the previous version as a paired difference per S1, with its standard error. A change
  is kept when it gains more than two standard errors on the full training set; on development subsets, differences
  below 0.0005 are treated as noise.
- The test-like validation adds copies of 90% of the unmatched training records, which brings the unmatched share to
  the test's 40%; the competition features are recomputed and both matchers score again out of fold. It predicted the
  v2 leaderboard score within 0.0006.
- A fair leave-one-country-out test (train on the US, score India, and the reverse, with the word statistics that need
  labels rebuilt from the training country only) stands in for France.

### 2.3 Challenges we faced and how we solved them

#### 1. A metric that punishes wrong merges

F0.5 counts precision twice, and a singleton loses its whole score on one wrong link. A fixed probability threshold
treats every S1 the same, whatever its other candidates. We choose each S1's links with the expected-F0.5 rule of
Section 4, where "no link" is always one of the options, and we pick the rule's settings on out-of-fold scores and
confirm them on the test-like set. The final version links with a pair precision of 0.99912 out of fold.

#### 2. Near-copies that look like true matches

A planted near-copy has the S1's name and street and differs in one detail. Plain similarity scores rate it as high as
a true match. We built features on the direction of each change: the signed house-number difference with a +1 to +25
bucket, the signed difference of the one run that changed in a composite number, whether a legal form was added,
dropped or changed, which short name tokens were substituted, and how faithfully the raw spelling was copied. On a
development subset the signed-shift and legal-direction features gained +0.00073, short tokens +0.00024 and the
raw-text features +0.00056 on the test-like set. The final output shows no bulk linking of near-copies: among linked
US pairs on the same street, shifts of +1 to +25 are as frequent as shifts of -25 to -1, as among true training pairs,
and legal-family changes appear at the true-pair rate (6.2 against 6.4 per 1,000 links).

#### 3. House numbers the first parser could not read

Indian prefixes and composite numbers, French bracketed numbers, five-digit numbers mistaken for postcodes, and
"Saint-X" read as a street hid the very detail that separates a near-copy from a true match. We rewrote the parser:
composite numbers are kept whole and compared run by run, units are a separate field, and a postcode is accepted only
when another component is the street. Parse rates rose from 47/46/42% to 91/84/82% for Indian S1/S2/S3 records, from
89/78/79% to 100/87/88% for the US, and from 98.6/90.5/90.8% to 99.4/93.0/93.2% for France; together with the
composite and unit features this gained +0.00049 on a development subset.

#### 4. A test set with more distractors than training

40% of test records match no S1 against 26% in training, and the test's distractors come in clusters. A model tuned on
training scores would link too much. We built the test-like validation to measure this, gave the negatives of
unmatched records a weight of 1.9 in training, and removed the per-S1 count features that drift between training and
test (adversarial validation AUC 0.737). Removing the counts gained +0.00023 on the test-like set, and the raw-text
features with the extra weight gained +0.00083 on the full training set's test-like view.

#### 5. A country with no labels

France is 15% of the test S1 and never appears in training, so no French decision can be learned from labels or
measured offline. We added label-free evidence measured per country on each split's own candidates: the word roles,
and a filler rule based on the legal-form evidence of Section 2.1. We blanked the label-based word statistics for 40%
of the S1 in training so the matchers learn to work without them, and we used the leave-one-country-out test as the
stand-in for France. There the word roles and the filler rule raised the held-out US from 0.98659 to 0.98874 and the
held-out India from 0.97191 to 0.97894. On the leaderboard the word roles gained +0.0007, all of it attributable to
France, and the filler rule another +0.00063. After v7 France has the same number of matches per S1 as the labelled
countries (France 3.362, US 3.394, India 3.376).

#### 6. Records nobody can resolve

A name-only record whose name belongs to several S1 of the same country carries no evidence of its owner. We measured
the ceiling this creates (0.99552), tested whether the owner could be guessed from the same-name S1's other records
(it could not), and leave these records unlinked; the expected-F0.5 rule does so by itself because each candidate gets
a low probability. Every result is reported with the ceiling and the gap to it, so we knew how much of the gap was
still reachable.

#### 7. Millions of pairs and a limited GPU budget

The training set has 17.36 million candidate pairs, too many for transformer models in the time we had. Most pairs are
already decided with near certainty: 14.3% of the training pairs hold 99.1% of the expected errors. The transformers
read only that uncertain band, and each fold trains within a fixed time budget. Stage 4 lowered the log loss on the
band from 0.1571 to 0.1469 and gained +0.00064 out of fold over stage 3; on the public leaderboard it moved the score
from 0.990684 to 0.991434.

#### 8. A record belongs to one business only

Each pair is scored on its own, so a record with two plausible S1 can get a high score on both. We assign every record
to its best S1 before selecting links, give the stacks query-level competition features, and divide a pair's score by
the record's total score when that total exceeds 1. Adding the competition of the two strongest scores to stage 4
(v13) gained +0.00009 over v11 (z = 8.6).

#### 9. A text model that fails quietly on an unseen country

Our first attempt for France (v12a) moved a fine-tuned cross-encoder to French pairs. It passed the
leave-one-country-out check but lowered the public score by 0.000198, because the check held out only the small
combiner, not the cross-encoder itself (details in Section 5). The final France step (v14) uses no text model, and its
check evaluates the exact transformation with France's own decision rule on both labelled countries. It changed 968 of
259,452 French rows and raised the public score from 0.991502 to 0.991514.

#### 10. The size of the search

Comparing all 1.73 million test S1 with all 9.97 million records would mean 1.73 x 10^13 pairs. We search inside
(country, state) partitions, merge states that true pairs often cross with a rule learned from the training data, and
cap the work per search task so that no worker runs far longer than the others. The result is 16.3 million candidate
pairs (a reduction ratio of 0.99999906); on the training set blocking and the prefilter keep 99.4% of the true
matches, and the CPU part of the pipeline runs in 3 hours 13 minutes on 48 vCPU.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** character 3-gram TF-IDF of the normalised name and address (3-grams hashed into a fixed
  vocabulary, weights computed on the split itself, name and address halves weighted equally). The search is
  partitioned by (country, state). States that true pairs often cross (for example Telangana and Andhra Pradesh) are
  merged by a rule learned from the training pairs, and records without a recognised state search their whole
  country. Country is an open set of labels: France goes through the same code with no special case.
- **Candidates kept:** the top 50 S1 per record with a score of at least 0.7 times the record's best score. A LightGBM
  prefilter then keeps the pairs needed for 99.95% recall of true pairs, and the best-ranked candidate is always kept.
  `candidate_pairs.tsv` is exactly this prefiltered set. No later stage adds or removes pairs: the matchers, the
  cross-encoders and the stacks all score these pairs, and every matched id is one of them.
- **Candidate pairs generated:** test: 51.96 million retrieved, **16,294,525** after the prefilter, for 1,732,544 S1
  and 9,969,589 records (1.63 candidates per record, 9.4 per S1). All possible pairs would be 1.73 x 10^13
  (6.72 x 10^12 within countries), so the reduction ratio is 0.99999906 (0.99999758 within countries). Training: 42.75
  million retrieved, 17.36 million after the prefilter.
- **How you ensured true matches were not lost:** state merges learned from data, a country-wide search for records
  without a state, a prefilter threshold set by recall, and the always-kept top candidate. On the training set
  blocking keeps 99.411% of true pairs (US 99.509%, India 99.263%), and 99.361% after the prefilter. On a development
  subset we classified the misses: 68% ranked beyond the top 50 (median rank 189), 23% were cut by the score ratio, 5%
  fell outside the record's state partition and 3% shared no 3-gram with their S1. Most are name-only records that keep
  one core word plus generic words, or brand-like names whose address carries the S1's composite number. An extra
  search on rare shared house numbers recovered a few of them but did not change F0.5 on the full data (paired
  -0.00003), so it is left off.

---

## 4. Matching Model

**Features used** (about 115 in stage 1 and 125 in stage 2, all computed on the pair and its candidate list):

- Name features: fuzzy string similarities (ratio, token sort and set, partial, Jaro-Winkler, concatenated names),
  word overlap weighted by rarity, the extra and missing words with their log-odds of being a true match (learned per
  fold from labels) and label-free marker statistics, alias and website-name matches, the legal form (conflict,
  equality, exact equality, and direction: added, dropped or family changed), short tokens missing, substituted at one
  edit or added, and how common the name is.
- Address features: TF-IDF similarities and their ranks, word overlap, street and city similarity, state agreement,
  house number (equality, edit distance, log and signed differences, absolute and relative, a +1 to +25 shift bucket),
  the main composite number (equality, digits-only equality so that "13-59" equals "1359", shape, signed difference of
  the single run that differs, extension or truncation), the signed difference of the last run, the overlap of all
  digit runs, and unit numbers (agreement and signed difference).
- Raw-text features, computed on the raw strings relative to the S1: the record's letter case and whether it equals
  the S1's, accents added or removed, extra double spaces, punctuation added or removed, raw name equality (as is, up
  to the legal form, ignoring case), and the address's letter case and component equality with digits masked.
- Word-role features (`src/roles.py`): for each country and word, among best-ranked candidate pairs on the same street
  whose house numbers are equal or shifted by +1 to +25, the share that keep the same address when the name change
  involves that word, kept apart for additions, deletions and replacements. Each share is compared with the country's
  same-address share among one-word typo replacements (Jaro-Winkler at least 0.85), which is the rate of pure noise
  and does not depend on how many distractors a split has. Every changed word is also scored in the other role it can
  play (added versus replacing a word), because the same word can be a near-copy suffix when added and a noise filler
  when it replaces a word (US "partners", French "groupe"). These statistics use no labels and are computed on each
  split's own candidates, leaving out the row being scored.
- Other: retrieval score, rank and margins; per-record competition statistics; in stage 2, competition features from
  the out-of-fold stage-1 scores (per record and per S1), and S1-consensus features (confident same-name records, and
  confident records that agree or conflict with this record's house number and street).

**Model type:** a stack of gradient-boosted trees and cross-encoders. Section 1.1 gives the purpose of each model;
the table below gives how each one is built and trained.

| component | details | licence, size |
|---|---|---|
| prefilter, stage-1 and stage-2 matchers | LightGBM binary classifiers, 5 folds split by S1, test scores averaged over the fold models; negatives of records that match no S1 weighted 1.9; the label-based word log-odds blanked for a hash-chosen 40% of S1 groups in training so the matchers also work for unseen vocabulary; per-S1 in-degree and rank counts left out because they drift with the test's distractors (adversarial validation AUC 0.737) | LightGBM (MIT) |
| character cross-encoder (stage 3) | 4-layer transformer, width 256, trained from scratch; reads record name, record address, S1 name and S1 address as characters plus hashed character pairs and triples (2^17 buckets each) with positions counted within each part; two fold models split by S1, test pairs get their mean; at most 2 epochs within a 24-minute training budget per fold, batch 512, AdamW learning rate 7e-4, distractor negatives weighted 1.9 | PyTorch (BSD-3), 70.35M parameters (67M of them in the two hashed tables, about 3.3M in the transformer layers), no pretrained weights |
| stage-3 stack | LightGBM (31 leaves, at least 500 rows per leaf) on the logits of the stage-1, stage-2 and cross-encoder scores, their query-level competition (rank and margin over the record's best other S1) and five simple record relations (house number equal, shifted or missing, composite number, legal-form relation, core-name and street equality), trained out of fold | LightGBM (MIT) |
| v9 matchers | the stage-1/2 matchers retrained with the cross-encoder score, its rank and its margin as features (blanked for 30% of S1 groups in training); their stage-2 score feeds stage 4 | LightGBM (MIT) |
| pretrained cross-encoders (stage 4 inputs) | `FacebookAI/xlm-roberta-base` (revision `e73636d4`), `FacebookAI/xlm-roberta-large` (revision `c23d21b0`, two runs), `intfloat/multilingual-e5-base` (revision `d1287505`); input `<s> record name \| record address </s></s> S1 name \| S1 address </s>` with name and address cut to 32 and 48 tokens; one output score, weighted binary cross-entropy, AdamW with linear warm-up and decay, bf16 mixed precision, at most one epoch within a fixed time budget per fold on one L40S (base models 35 minutes, large 30 minutes, second large run 45 minutes with seed 2 and the S1 record read first) | MIT; 279M, 561M and 278M parameters |
| stage-4 stack | LightGBM of the stage-3 size, trained out of fold on the uncertain pairs only: the stage-3 features, the stage-3 score, each transformer's logit, the query-level competition of their mean, the v9 stage-2 score, and the query-level competition of the two strongest scores (for the stage-3 score its margin over the record's best other S1, its rank and the other S1s' total score; for the v9 score its margin); three seeds averaged; the final score is divided by the record's total score when that total exceeds 1 | LightGBM (MIT) |

The transformers read only the uncertain band: pairs whose stage-3 out-of-fold score lies in [0.002, 0.998]. That is
2,479,247 training pairs (14.3% of all training pairs, holding 99.1% of the expected errors) and 1,554,017 test pairs
of the labelled countries (US 721,417, India 832,600). Pretraining gives these models knowledge of names, places,
spelling variants and scripts (Latin, Devanagari and other Indian scripts) that the character model lacks. Pairs
outside the band keep their stage-3 score.

**Same-address filler rule (`src/fillers.py`, applied before the links are chosen).** From each split's own
best-ranked candidate pairs at the same address (same street up to a typo or a stray letter, equal house number, no
address or unit number that the S1 lacks) whose names differ by one replaced word, we count for each country and new
word how often the legal form is kept. A word with at least 30 such pairs and at most 5% legal-form changes is a noise
filler of that country. A candidate at the same address whose name is the S1 name with one word replaced by a filler,
or with a filler added at the end, and with the same legal form, gets a probability of at least 0.95, unless the
record qualifies for two S1 (several businesses at one address). No labels, word lists or country names are involved.
On the training countries the rule finds exactly the known fillers (`center`, `services`, `service`, `partners`, plus
two Indian transliteration variants) and leaves the out-of-fold score unchanged (22 pairs raised, all true). With one
country held out it helps (US held out +0.00026, India held out +0.00003). On the test set it finds the French fillers
by itself, including the three words that double as near-copy suffixes (`groupe`, `developpement`, `france`).

**Countries without training labels (`src/scripts/v14_france.py`).** France is the only such country in this test set,
but nothing here is specific to it: every step decides by whether a pair's country is in the set of training
countries, never by its name, so any other new country label takes exactly the same path. France is scored by the
stage-2 matchers (which carry the label-free word roles and the blanked word log-odds) and the filler rule. The
cross-encoder stacks are not applied to it, because they have never seen a French pair and no French label exists to
check them. The final step adds the query-level competition that the stacks give the labelled countries. A French
record with two plausible S1 (18% of French S1 share an address with another S1, 34% share a name) could keep a score
of 0.8 on its best S1 next to 0.5 on another, and the decision rule would treat 0.8 as its probability. Two
transformations of the stage-2 score use nothing but the score and the record's own candidate list:

- `qn@s`: the score divided by max(1, the record's total score over its S1 candidates), then the same decision rule
  at probability scale s, for s in 1.0, 1.1, 1.2 and 1.3;
- `lr(LOCO)`: a three-weight logistic recalibration of the uncertain pairs on the stage-2 logit, its margin over the
  record's best other S1 and the logit of the normalised score, fitted on US and India out-of-fold scores and checked
  with one country held out.

The check: each candidate transformation is applied to the US and Indian out-of-fold stage-2 scores with France's own
decision rule and compared with the plain score, paired per S1, on the plain out-of-fold set and on the test-like set.
A transformation is eligible only if it gains more than two standard errors in each labelled country on both sets;
the eligible one with the best blended gain (out of fold + 0.31 x (test-like - out of fold), the blend that matched
the v2 to v3 leaderboard change) is applied. If none is eligible, nothing changes. Before anything is written, the
script re-derives the base output's 259,452 French rows from the stage-2 score, the filler rule and the decision rule
and requires all of them to match, and it copies the US and Indian rows unchanged and verifies them.

The check chose `qn@1.1`, the only candidate that gained more than two standard errors in both countries on both
views: out of fold US +0.000035 (s.e. 0.000009) and India +0.000047 (0.000011), and on the test-like view US +0.000039
and India +0.000051. `lr(LOCO)` had the same blended gain (+0.000041), but its test-like gain in India (+0.000029, s.e.
0.000015) fell short, and larger scales traded precision for recall (full table in Appendix B).

**Threshold selection method:** there is no single threshold. Each record is first assigned to the S1 with its highest
score. Then, for each S1, its records are sorted by probability, and we keep the top k records for the k that
maximises the expected F0.5, treating each record as correct with its probability; k = 0 (no link) is always an
option, which protects singletons. Two settings shape this rule: a small probability that a true match was missed
(0 or 0.02) and a probability scale (0.8 to 1.1) that makes the rule more or less willing to link. Both are chosen on
out-of-fold scores and checked on the test-like set, where the best setting is flat. For the final stage 4, scales 0.9,
1.0 and 1.1 score the same on the test-like set (0.99200); scale 1.1 has the largest out-of-fold gain (+0.00009 over
v11, z = 8.6) and is the submitted setting. Countries without labels use the rule tuned on the stage-2 scores (scale
1.1), applied to the normalised stage-2 score chosen by the v14 check.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.99218** on the full training set out of fold (all 2,206,821 training S1, US and India,
  each scored by models that never saw it; v11 0.99209 plus a paired gain of +0.00009). Public leaderboard of the final
  version: **0.991514**. For reference, the ceiling on the training set is 0.99552.

Scores by version. Out-of-fold and test-like numbers are on the full training set; gains are paired per S1.

| version | change | out of fold | test-like | public LB |
|---|---|---|---|---|
| baseline | TF-IDF blocking, prefilter, two LightGBM matchers, 2 folds | 0.98896 | 0.98553 | |
| v2 | signed number shifts, house-number parser, composite and unit features, short tokens, 5 folds, S1-consensus features, word log-odds dropout, drifting counts removed | 0.99036 (+0.00140) | 0.98849 (+0.00295) | 0.987916 |
| v3 | raw-text features, distractor weight 1.9 | 0.99061 (+0.00025) | 0.98932 (+0.00083) | 0.988272 |
| v6 | label-free word roles in both roles a word can play | -0.00005 vs v3 (development subset) | | 0.988977 |
| v7 | same-address filler rule | unchanged (22 pairs raised, all true) | | 0.989604 |
| v8 | character cross-encoder + stage 3 | 0.99142 (+0.00078 vs v6; US +0.00018, India +0.00169) | | 0.990684 |
| v9 | matchers with cross-encoder features (stage-4 input only) | 0.99122 (+0.00058 vs v6) | +0.00138 vs v6 | not submitted |
| v10 | XLM-R base and large on the uncertain band + v9 score, stage 4 | 0.99206 (+0.00064 vs v8; US +0.00037, India +0.00106) | | 0.991434 |
| v11 | + multilingual-E5 base | 0.99209 | | not submitted |
| v12a | cross-encoder transferred to France | same (labelled rows unchanged) | | 0.991236 (rejected) |
| v13 | second XLM-R large, stage-3 and v9 query competition, 3 seeds, normalised scores, scale 1.1 | 0.99218 (+0.00009 vs v11, z 8.6) | 0.99200 | 0.991502 |
| **v14 (final)** | v13 + checked competition-aware decode for countries without labels | same (labelled rows unchanged) | | **0.991514** |

The versions for France (v4 to v7) cannot be scored offline, so their yardstick is the fair leave-one-country-out
test: US held out 0.98659, then 0.98848 with the word roles and 0.98874 with the filler rule; India held out 0.97191,
0.97891 and 0.97894, with the in-country scores unchanged. On the leaderboard v6 gained +0.0007 over v3, all of it
attributable to France, and v7 added +0.00063 with 13,628 more links (13,696 new, 68 dropped); at about 0.08 F0.5 per
added link, about 98% of the added links were correct. The public gains of v8 and v10 were 1.26 and 1.19 times the
gains projected from out-of-fold scores by country share, consistent with a test that rewards precision more than the
training data does.

Effect of the final step (v14) on the test set: the US and Indian rows are identical to v13. In France, 968 of 259,452
rows changed, with 991 links removed and 11 added; no S1 was emptied and none newly linked, so 0.11% of French links
changed. The public score moved from 0.991502 to 0.991514.

- **Common false positives (wrong merges):** near-copies whose change is hidden in a composite or unit number
  ("Unit # 103" to "# 112", "5-4-187/3&4" to "3&5"), near-copies that change one short name token at the same address,
  and records of the same business name at another address. Checks of the output that need no labels show that
  near-copies are not linked in bulk: among linked US pairs on the same street, house-number shifts of +1 to +25 and
  -25 to -1 are as frequent as among true training pairs (near-copies only shift upward), and legal-family changes
  occur at the true-pair rate (6.2 against 6.4 per 1,000 links).
- **Common false negatives (missed matches):** name-only records whose name belongs to several S1 (unresolvable, see
  Section 2.1), true matches with large number noise, and records that blocking never retrieved (about 0.6% of true
  pairs).

What we tried and did not keep (each measured paired per S1 on the full data or a full-density subset):

| experiment | result | decision |
|---|---|---|
| training on copies of unmatched records | test-like +0.00283 but out of fold -0.00101; the model learns the copies, not the distractors | no |
| distractor-count features (rejected and empty-address neighbours) | test-like -0.00043 | no |
| house-number search channel in blocking | full data -0.00003 | no |
| larger trees (255 leaves) | -0.00001 | no |
| prior-shift correction before choosing links | precision and recall changes cancel | diagnostic only |
| self-training on the held-out country | +0.0014 to +0.0016 on the held-out country, but it reinforces its own wrong links and needs pseudo-labels | no |
| averaging several trained variants | about +0.0001 | no |
| same-address one-word swap rule for words unknown to the label-based statistics | held-out US -0.00036 to -0.00253 | no |
| v12a: XLM-R base transferred to France with a logistic combiner | leave-one-country-out +0.00027 / +0.00149, but public -0.000198 | no (reason below) |
| stage 5 (competition of the stage-4 score) | did not beat stage 4 with normalised scores | no |
| name-only tie rule for France (drop links of empty-address records to one of several same-name S1) | on the labelled countries the dropped links were 78% (US) and 83% (India) true; F0.5 -0.000054 / -0.000079 | no |

Why v12a failed: its leave-one-country-out check held out only the small logistic combiner, while the cross-encoder
had been fine-tuned on both labelled countries, so neither held-out country was new to it. Warning signs that need no
labels were visible: 55.7% of French band pairs had a positive cross-encoder logit against 20.6% in the labelled
countries, and its rank correlation with the stage-2 score was 0.37 against 0.76. Mapping the French scores onto the
labelled countries' distribution corrected their level but not their order. v14 uses no text model, and its check
evaluates the exact transformation, with France's decision rule, on both labelled countries.

---

## 6. Conclusion

A loss budget per error type showed that most of the remaining error came from generator-planted near-copies and from
house numbers the parser could not read. Features that expose the direction of each change, a parser for every
number format we met, and cross-encoders focused on the uncertain pairs gave the largest gains, from 0.98896 to
0.99218 out of fold and 0.991514 on the public leaderboard. Every change was validated on a test-like copy of the
training data and compared per S1 with the previous version, and every rule for the unseen country had to help on the
labelled countries first. The measured ceiling of 0.99552 shows how little of the remaining gap can still be reached
from the records alone.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains the complete pipeline and nothing else:

- `src/`: the Python package (normalisation, blocking, features, word roles, filler rule, LightGBM training, link
  selection, evaluation, the character and pretrained cross-encoders, the stack features) and `src/scripts/`, which
  holds the nine-step runner `reproduce.sh` and the Python scripts of steps 2 and 4 to 9.
- `README.md`: set-up on a new machine, the command, what each step runs and writes, run times, where the output is
  saved and what to look for in the logs.
- `requirements.txt`: pinned versions (Python 3.12; numpy, pandas, pyarrow, scipy, scikit-learn, lightgbm, rapidfuzz,
  anyascii, torch 2.14.0, transformers 4.57.1, tokenizers 0.22.1).

Entry point: `bash src/scripts/reproduce.sh <student_resource>` runs, in order, the base pipeline
(`python -m src.run all`, which writes `candidate_pairs.tsv`), the character cross-encoder and stage 3, the retrained
matchers, the uncertain band, the four transformer fine-tunings, the reference and final stage-4 stacks, the decoder
scale and the step for countries without labels, which writes `matching_results.tsv` and copies the candidate set.

Measured run times: the CPU base pipeline takes 3 h 13 min on 48 vCPU (training 2 h 02 min including 42 min of
retrieval; test 1 h 11 min). The later steps ran on one AWS ml.g6e.16xlarge (64 vCPU, one NVIDIA L40S): character
cross-encoder and stage 3 105 min, v9 104 min, band export about 10 min, XLM-R base about 1 h, multilingual-E5 about
40 min, XLM-R large about 2 h, the second XLM-R large run about 1 h 40 min, each stage-4 stack about 20 to 25 min, the
decoder-scale step about 5 min and the final step about 10 to 15 min.

Storage: the data, the intermediate files and the outputs stayed on the instance's local disk. The pipeline reads and
writes no cloud storage (no S3 bucket), and nothing is sent anywhere.

Reproducibility check: we re-ran steps 8 and 9 of this package on SageMaker from the saved stage-4 scores of step 7.
It reproduced the submitted v13 file (md5 `a9bae0849410ab0fd09973d525a09608`), the v14 file and the candidate set byte
for byte in 11 minutes. A full rerun from the raw files runs the same code with the same settings and lands very close
to the submitted file but not byte for byte, because the cross-encoders train for a fixed time per fold (so the number
of steps depends on the GPU), GPU arithmetic is not exactly repeatable, and our base models were trained step by step
on cached features of earlier versions. The README explains this in more detail.

### B. Additional Results

v14 gate on the labelled countries (Section 4). Each row is a candidate transformation of the stage-2 score, decoded with the unlabelled-country rule and compared with the plain stage-2 score paired per S1: mean F0.5 change (standard error) on the plain out-of-fold set and on the test-like set.

Plain stage-2 score with this decoding rule: out-of-fold F0.5 0.99064, test-like 0.99026.

| transformation | blended gain | India out of fold | India test-like | US out of fold | US test-like | eligible |
|---|---|---|---|---|---|---|
| qn@1.0 | +0.000006 | -0.000036 (0.000018) | +0.000022 (0.000020) | +0.000017 (0.000013) | +0.000032 (0.000013) | no |
| qn@1.1 | +0.000041 | +0.000047 (0.000011) | +0.000051 (0.000011) | +0.000035 (0.000009) | +0.000039 (0.000009) | yes |
| qn@1.2 | +0.000030 | +0.000056 (0.000016) | +0.000011 (0.000017) | +0.000027 (0.000011) | +0.000010 (0.000011) | no |
| qn@1.3 | -0.000036 | +0.000001 (0.000019) | -0.000097 (0.000021) | -0.000029 (0.000012) | -0.000069 (0.000013) | no |
| lr(LOCO) | +0.000041 | +0.000052 (0.000014) | +0.000029 (0.000015) | +0.000039 (0.000010) | +0.000037 (0.000010) | no |

Chosen transformation: `qn@1.1`.
Logistic recalibration weights fitted on both labelled countries (logit, margin, normalised logit, intercept): 0.036, 0.110, 0.958, 0.018.

Fair leave-one-country-out (the stand-in for France; the label-based word statistics of the held-out country rebuilt
from the other country only). Training on one country and scoring the other loses 0.008 to 0.020 F0.5, mostly through
wrong links on singletons. Blanking the label-based word log-odds for 40% of S1 groups in training recovers part of it
(US held out 0.98168 to 0.98437, India held out 0.96586 to 0.97166) at no in-country cost, and is used.

Development subset results that shaped the base pipeline (every record of four states, paired per S1): signed
house-number and legal-direction features +0.00073; the new parser with composite and unit features +0.00049; short
name tokens +0.00024; 5 folds instead of 2 +0.00091 (test-like +0.00103); dropping the drifting per-S1 counts +0.00023
on the test-like set; raw-text features +0.00019 out of fold and +0.00056 on the test-like set.

Final output (`output/`): 1,732,544 S1 rows (one per test S1), 1,632,788 with at least one match, 5,865,385 matched ids. Per country: India 3.385 matches per S1 (5.75% empty); US 3.396 matches per S1 (5.75% empty); France 3.358 matches per S1 (5.79% empty). Candidate set: 16,294,525 pairs (9.40 per S1, 1.63 per S2/S3 record; reduction ratio 0.99999906 against all S1 x S2/S3 pairs); every matched id is a candidate (validator PASS with `--check-ids`).

### C. Rule Compliance

- Data: only the provided training and test files. The normalisation tables (abbreviations, legal forms, state and
  region names) are hand-written and generic; a country without an entry simply searches country-wide. No external
  data, APIs, geocoding or lookups of any business.
- No language model labels, judges or matches any record. The test set is used only as unlabelled input (the word-role
  and filler statistics count patterns over the test's own candidate pairs, as retrieval does); there are no
  pseudo-labels and no training on the test set.
- Models: LightGBM trees (MIT), a character cross-encoder trained from scratch (PyTorch, BSD-3), and three pretrained
  encoders fine-tuned as pair classifiers: `FacebookAI/xlm-roberta-base` (279M parameters),
  `FacebookAI/xlm-roberta-large` (561M) and `intfloat/multilingual-e5-base` (278M), all MIT-licensed and far below the
  8B limit, loaded with Hugging Face `transformers` (Apache-2.0). Their weights are downloaded once at the pinned
  revisions in `src/ptce.py`, run locally and fine-tuned on the provided training pairs only.
- Country is an open set of labels. Nothing is filtered or one-hot encoded by country, no model or decision rule is
  keyed to a country name (only the hand-written normalisation tables are organised by country label), and every test
  S1, France included, has a row in the output. A country that is new to the pipeline has no entry in the region-name
  table, so its records are searched across the whole country, and the label-free word roles, filler words and the
  final transformation are measured on its own records. The hand-written abbreviation and legal-form tables cover the
  conventions found in the data (English, French and Indian); for another language the pipeline runs unchanged but
  those words are not normalised.
- Entity ids are used only to split folds and to write the output, never as features or blocking keys; row and file
  order are not used either.

### D. Terms used in this document

Entity resolution has its own vocabulary, and this document uses a few short forms. They are all explained here.

| term | meaning |
|---|---|
| S1, S2, S3 | Source 1 is the clean reference list of businesses. Source 2 and Source 3 are noisy lists whose records we match to Source 1. "An S1" means one Source-1 business; "a record" means one row of S2 or S3. |
| match, link | an (S1, record) pair that we output as the same business |
| singleton | an S1 with no true match in S2 or S3 (about 5.6% of the training S1). It scores 1.0 only if we link nothing to it. |
| precision, recall | precision is the share of our links that are correct; recall is the share of the true matches that we found |
| F0.5 | the challenge metric: F0.5 = 1.25 x P x R / (0.25 x P + R), with P the precision and R the recall. With beta = 0.5, precision counts twice as much as recall, so a wrong merge costs more than a missed match. F0.5 is computed for each S1 and then averaged over all S1 (a "macro" average). The challenge's own example: 2 correct links out of 3, with both true matches found, gives P = 0.667, R = 1.0 and F0.5 = 0.714. |
| expected F0.5 | the F0.5 we expect for a set of links when each link is right with the probability our model gives it. We use it to choose the links of each S1 (Section 4). |
| blocking, candidate pair | blocking is the fast first pass that picks, for every record, a short list of S1 worth comparing; each (S1, record) pair on that list is a candidate pair. Only candidate pairs can become matches. |
| TF-IDF, 3-gram | a text similarity that compares two strings through their overlapping 3-character pieces (3-grams), with rare pieces weighted more than common ones |
| prefilter | a small model that removes candidate pairs that are clearly not matches before the expensive models run |
| LightGBM | a gradient-boosted decision-tree library (MIT licence); all our matchers and stacks are LightGBM models |
| feature | a number computed for a pair that a model uses, such as a name similarity or a house-number difference |
| cross-encoder | a neural network that reads both records of a pair together and outputs one match score |
| pretrained transformer, fine-tuning | a language model already trained on large public text (here XLM-RoBERTa and multilingual-E5); fine-tuning continues its training on our training pairs so that it learns to score matches |
| stage, stack | our models run in stages; a later stage (a "stack") takes the earlier stages' scores as inputs and learns how to combine them |
| fold, out of fold (OOF) | the training S1 are split into groups called folds. Each model is trained on all folds but one and scores the one it did not see, so every training pair gets a score from a model that never saw it, just as test pairs do. |
| test-like validation | our out-of-fold evaluation after adding copies of unmatched training records, so that 40% of records match no S1, as on the test set (26% in the raw training data) |
| paired difference, s.e., z | to compare two versions we compute the F0.5 difference on every S1 and average it. The standard error (s.e.) says how uncertain that average is, and z = difference / s.e. We keep a change when z is above 2. |
| LOCO | leave-one-country-out: train on the US and score India, or the reverse. It stands in for France, which has no labels. |
| near-copy (distractor) | a record planted by the data generator to look like an S1 business while belonging to a different one, for example the same name with a house number a few units higher |
| legal form | the company-type word in a name: Inc, LLC, Pvt Ltd, SARL and so on |
| noise filler | a generic word that true matches add to a name or swap into it without changing the business (US "center", "services"; French "groupe", "fils") |
| label-free | computed without the ground truth, so it can be computed on the test set and for France as well |
| band (uncertain pairs) | the pairs whose stage-3 score lies between 0.002 and 0.998; only these are re-read by the transformers |
| logit | log(p / (1 - p)): a probability p put on an unbounded scale, which is easier for a model to combine |
| query-level competition | features that compare a record's score for one S1 with its scores for the other S1 it could belong to. A record belongs to at most one S1. |
| public and private leaderboard | the public leaderboard scores part of the test set during the challenge; the private leaderboard scores the rest and decides the final ranking |
| vN | a version of our solution (v7, v8, ...); Figure 2 in Section 2.2 shows what each one added |
