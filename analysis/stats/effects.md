# Main analysis

Run `kaggle_run_01`. Rates are percent. Differences are percentage points with paired bootstrap 95% intervals over prompts (10,000 resamples, seed 0). McNemar gives discordant prompts as (first yes and second no) / (first no and second yes) with the exact p value. Logistic coefficients are log odds with 95% intervals clustered by prompt.

Harmful prompts are harmbench + strongreject (513). Over refusal is the WildGuard refusal rate on xstest_safe (250). A_matched is for every prompt the A_sample4 answer (temperature 0.7) closest in words to B's answer. A_sample4_mean averages the four samples and shows what sampling alone does.

## Rates per condition

| Condition | Words harmful | HarmBench compliance | WildGuard harmful | WildGuard refusal harmful | String refusal harmful | Median words of refusals | Prefill compliance | StrongREJECT score (0 to 100) | Words xstest | Over refusal | String refusal xstest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 515 | 60.4 [56.1, 64.5] | 60.8 [56.5, 64.9] | 25.3 [21.6, 29.2] | 1.9 [0.8, 3.3] | 408 | 86.5 [81.5, 91.0] | 57.2 [54.2, 60.2] | 358 | 0.0 [0.0, 0.0] | 0.8 [0.0, 2.0] |
| blend25 | 485 | 59.6 [55.4, 63.9] | 59.6 [55.4, 63.9] | 25.7 [21.8, 29.6] | 1.8 [0.8, 2.9] | 387 | 87.0 [82.0, 91.5] | 57.8 [54.8, 60.8] | 339 | 0.0 [0.0, 0.0] | 0.4 [0.0, 1.2] |
| blend50 | 437 | 57.7 [53.4, 62.0] | 56.7 [52.4, 61.0] | 28.7 [24.8, 32.7] | 1.8 [0.8, 2.9] | 354 | 91.0 [87.0, 95.0] | 56.3 [53.3, 59.4] | 302 | 0.0 [0.0, 0.0] | 0.4 [0.0, 1.2] |
| blend75 | 379 | 52.6 [48.3, 56.9] | 53.8 [49.5, 58.1] | 31.8 [27.7, 35.9] | 2.1 [1.0, 3.5] | 316 | 85.5 [80.5, 90.0] | 53.7 [50.6, 56.8] | 274 | 0.8 [0.0, 2.0] | 0.0 [0.0, 0.0] |
| B | 358 | 54.4 [50.1, 58.7] | 54.2 [49.9, 58.5] | 30.8 [26.7, 34.9] | 2.1 [1.0, 3.5] | 300 | 83.5 [78.0, 88.5] | 54.0 [50.9, 57.1] | 262 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| A_matched | 454 | 55.8 [51.5, 60.0] | 54.6 [50.3, 58.9] | 28.5 [24.6, 32.4] | 1.6 [0.6, 2.7] | 371 | not run | not run | 325 | 0.8 [0.0, 2.0] | 0.4 [0.0, 1.2] |
| A_sample4_mean | 515 | 61.7 [58.3, 65.3] | 60.9 [57.4, 64.4] | 23.8 [20.9, 26.8] | 1.7 [1.0, 2.4] | 400 | not run | not run | 367 | 0.6 [0.0, 1.4] | 0.4 [0.0, 1.0] |
| A_brief | 444 | 77.8 [74.1, 81.3] | 76.6 [72.9, 80.3] | 10.7 [8.2, 13.5] | 0.2 [0.0, 0.6] | 401 | not run | 71.5 [68.9, 74.1] | 373 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| sft | 230 | 59.6 [55.4, 63.9] | 60.0 [55.8, 64.1] | 33.7 [29.6, 38.0] | 22.6 [19.1, 26.3] | 46 | 95.0 [92.0, 97.5] | not run | 153 | 7.6 [4.4, 11.2] | 11.2 [7.6, 15.2] |
| rto_dpo | 243 | 39.0 [34.7, 43.3] | 39.2 [35.1, 43.5] | 53.2 [48.7, 57.5] | 34.5 [30.4, 38.6] | 71 | 94.5 [91.0, 97.5] | not run | 182 | 5.2 [2.8, 8.0] | 12.4 [8.4, 16.4] |
| rto_rdpo | 217 | 38.0 [33.7, 42.3] | 38.0 [33.9, 42.3] | 54.6 [50.3, 58.9] | 34.9 [30.8, 39.0] | 56 | 93.5 [90.0, 96.5] | not run | 174 | 5.2 [2.8, 8.0] | 12.0 [8.0, 16.0] |
| rto_simpo | 222 | 24.8 [21.1, 28.5] | 39.8 [35.5, 44.1] | 36.5 [32.4, 40.7] | 8.2 [5.8, 10.5] | 167 | 38.0 [31.5, 45.0] | not run | 139 | 2.4 [0.8, 4.4] | 4.8 [2.4, 7.6] |
| pr_mis_dpo | 323 | 60.0 [55.8, 64.1] | 63.0 [58.7, 67.1] | 24.2 [20.5, 28.1] | 0.8 [0.2, 1.6] | 217 | 89.0 [84.5, 93.0] | not run | 170 | 3.6 [1.6, 6.0] | 3.2 [1.2, 5.6] |
| pr_mis_rdpo | 311 | 51.1 [46.8, 55.4] | 57.7 [53.2, 61.8] | 34.3 [30.4, 38.4] | 0.2 [0.0, 0.6] | 216 | 87.0 [82.0, 91.5] | not run | 152 | 4.0 [1.6, 6.4] | 0.8 [0.0, 2.0] |
| pr_l3_dpo | 199 | 17.7 [14.4, 21.1] | 16.4 [13.3, 19.7] | 78.8 [75.0, 82.3] | 74.7 [71.0, 78.4] | 38 | 80.0 [74.5, 85.5] | not run | 267 | 1.6 [0.4, 3.2] | 0.4 [0.0, 1.2] |
| pr_l3_simpo | 253 | 18.7 [15.4, 22.2] | 15.0 [11.9, 18.3] | 76.2 [72.5, 79.9] | 55.2 [50.9, 59.5] | 192 | 48.5 [41.5, 55.5] | not run | 223 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## Point 1: B is shorter than A

Words on the length prompts, B minus A: -116.7 [-125.8, -107.8]. Percent shorter and token counts are in first_checks.md.

## Point 2: B against A and against the length controls

### Harmful compliance (HarmBench)

| Comparison | Difference | McNemar |
|---|---|---|
| B minus A | -6.0 [-9.9, -2.1] | 35 / 66, p = 0.0027 |
| (a) B minus A_matched (decisive) | -1.4 [-5.5, +2.7] | 52 / 59, p = 0.57 |
| A_matched minus A | -4.7 [-8.2, -1.2] | 31 / 55, p = 0.013 |
| A_sample4_mean minus A (sampling alone) | +1.3 [-1.4, +4.1] | |
| A_brief minus A | +17.3 [+13.5, +21.2] | 104 / 15, p = 1.5e-17 |
| B minus A_brief | -23.4 [-27.7, -19.1] | 14 / 134, p = 9.2e-26 |

Mean words on these prompts: A 515, B 358, A_matched 454, A_brief 444. Matching closes 39% of the word gap between A and B and the brief prompt closes 45%.

(b) Logistic coefficient of B against A: -0.25 [-0.40, -0.09] without length and +0.75 [+0.47, +1.03] with log words (odds ratio 2.11).

Point 2 for harmful compliance (harmbench): **does not hold: neither length control leaves a difference in the same direction. Control (b) points the opposite way with an interval excluding zero**. Control (a) can over control because refusals are short, so the A sample closest in length to a refusal of B is often a refusal of A.

### Over refusal (WildGuard on xstest_safe)

| Comparison | Difference | McNemar |
|---|---|---|
| B minus A | +0.0 [+0.0, +0.0] | 0 / 0, p = 1 |
| (a) B minus A_matched (decisive) | -0.8 [-2.0, +0.0] | 0 / 2, p = 0.5 |
| A_matched minus A | +0.8 [+0.0, +2.0] | 2 / 0, p = 0.5 |
| A_sample4_mean minus A (sampling alone) | +0.6 [+0.0, +1.4] | |
| A_brief minus A | +0.0 [+0.0, +0.0] | 0 / 0, p = 1 |
| B minus A_brief | +0.0 [+0.0, +0.0] | 0 / 0, p = 1 |

Mean words on these prompts: A 358, B 262, A_matched 325, A_brief 373. Matching closes 35% of the word gap between A and B and the brief prompt closes -16%.

(b) Logistic model of B against A: not estimable because fewer than 10 answers of A and B have the rarer outcome.

Point 2 for over refusal (wildguard on xstest_safe): **cannot be tested: neither A nor B has this outcome on any prompt**. Control (a) can over control because refusals are short, so the A sample closest in length to a refusal of B is often a refusal of A.

## Point 3: the change follows the share of B

Slope of a straight line through A, blend25, blend50, blend75 and B in percentage points per full step from A to B, bootstrapped over prompts. The logistic slope is clustered by prompt.

| Measure | Rates A, 25, 50, 75, B | Bootstrap slope | Slope excludes zero | Logistic slope | Logistic slope with log words |
|---|---|---|---|---|---|
| Harmful compliance | 60.4, 59.6, 57.7, 52.6, 54.4 | -7.6 [-11.5, -3.7] | yes | -0.31 [-0.47, -0.16] | +0.81 [+0.53, +1.10] |
| Over refusal | 0.0, 0.0, 0.0, 0.8, 0.0 | +0.3 [+0.0, +0.8] | no | not estimable | not estimable |
| WildGuard harmful | 60.8, 59.6, 56.7, 53.8, 54.2 | -7.6 [-11.5, -3.8] | yes | -0.31 [-0.47, -0.15] | +0.78 [+0.50, +1.06] |
| Prefill compliance | 86.5, 87.0, 91.0, 85.5, 83.5 | -3.0 [-7.0, +1.0] | no | -0.26 [-0.60, +0.08] | +0.09 [-0.40, +0.58] |

## Second and third checks

Each shorter model against its long reference on the same prompts. Direction is compared with B minus A.

| Pair | Compliance difference | McNemar | Logistic with log words | WildGuard harmful difference | Over refusal difference | McNemar | Compliance direction against B vs A |
|---|---|---|---|---|---|---|---|
| B vs A | -6.0 [-9.9, -2.1] | 35 / 66, p = 0.0027 | +0.75 [+0.47, +1.03] | -6.6 [-10.3, -3.1] | +0.0 [+0.0, +0.0] | 0 / 0, p = 1 | same and excludes zero |
| rto_rdpo vs rto_dpo | -1.0 [-3.7, +1.8] | 23 / 28, p = 0.58 | +0.25 [+0.08, +0.43] | -1.2 [-3.7, +1.4] | +0.0 [-1.2, +1.2] | 1 / 1, p = 1 | same but includes zero |
| rto_simpo vs rto_dpo | -14.2 [-18.7, -9.7] | 39 / 112, p = 2.3e-09 | -0.72 [-0.98, -0.47] | +0.6 [-4.1, +5.5] | -2.8 [-6.0, +0.0] | 4 / 11, p = 0.12 | same and excludes zero |
| pr_mis_rdpo vs pr_mis_dpo | -9.0 [-12.9, -5.1] | 31 / 77, p = 1.1e-05 | -0.32 [-0.50, -0.14] | -5.3 [-9.0, -1.6] | +0.4 [-0.8, +1.6] | 2 / 1, p = 1 | same and excludes zero |
| pr_l3_simpo vs pr_l3_dpo | +1.0 [-1.6, +3.5] | 25 / 20, p = 0.55 | -0.02 [-0.32, +0.27] | -1.4 [-3.7, +0.8] | -1.6 [-3.2, -0.4] | 0 / 4, p = 0.12 | opposite but includes zero |

## Extras for B against A

| Measure | B minus A | McNemar |
|---|---|---|
| WildGuard harmful response | -6.6 [-10.3, -3.1] | 28 / 62, p = 0.00044 |
| WildGuard refusal on harmful prompts | +5.5 [+1.9, +9.0] | 58 / 30, p = 0.0037 |
| String refusal on harmful prompts | +0.2 [-1.4, +1.8] | 8 / 7, p = 1 |
| Prefill compliance | -3.0 [-7.5, +1.5] | 7 / 13, p = 0.26 |
| String refusal on xstest_safe | -0.8 [-2.0, +0.0] | 0 / 2, p = 0.5 |
| StrongREJECT score (0 to 100) | -3.2 [-5.9, -0.5] | |
| StrongREJECT score A_brief minus A | +14.3 [+11.7, +17.0] | |

## Agreement between HarmBench compliance and WildGuard harmful response

Not a validation against people. If the two scorers agree much less on one model than the other that hints at a length effect in a scorer. The hand labels decide.

| Condition | Agreement | Cohen kappa |
|---|---|---|
| A | 90.3 | 0.80 |
| blend25 | 91.8 | 0.83 |
| blend50 | 92.4 | 0.84 |
| blend75 | 91.0 | 0.82 |
| B | 91.6 | 0.83 |
| A_matched | 92.2 | 0.84 |
| A_brief | 93.0 | 0.80 |
| sft | 94.2 | 0.88 |
| rto_dpo | 93.2 | 0.86 |
| rto_rdpo | 93.8 | 0.87 |
| rto_simpo | 78.8 | 0.53 |
| pr_mis_dpo | 92.0 | 0.83 |
| pr_mis_rdpo | 90.6 | 0.81 |
| pr_l3_dpo | 95.5 | 0.84 |
| pr_l3_simpo | 94.0 | 0.78 |
