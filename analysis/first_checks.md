# First checks

Run `kaggle_run_01`, code commit `82cf9f4`, started 2026-10-29 23:51:52 UTC. Intervals are paired bootstrap 95% intervals over prompts with 10,000 resamples.

## Length gaps

Words are whitespace words, tokens are each model's own tokens. A negative difference means the first model is shorter.

| Pair | Prompts | n | Long words | Short words | Word difference | % shorter (words) | % shorter (tokens) |
|---|---|---|---|---|---|---|---|
| B vs A | length | 300 | 375 | 259 | -116.7 [-125.8, -107.8] | 31.1 [29.2, 33.0] | 28.8 [26.3, 31.1] |
| rto_rdpo vs rto_dpo | length | 300 | 225 | 214 | -11.7 [-20.4, -4.3] | 5.2 [1.9, 8.8] | 4.2 [0.7, 7.8] |
| rto_simpo vs rto_dpo | length | 300 | 225 | 161 | -64.1 [-77.2, -52.0] | 28.5 [24.3, 32.5] | 19.7 [14.9, 24.5] |
| pr_mis_rdpo vs pr_mis_dpo | length | 300 | 228 | 200 | -28.0 [-38.4, -18.0] | 12.3 [8.1, 16.4] | 15.5 [10.9, 20.1] |
| pr_l3_simpo vs pr_l3_dpo | length | 300 | 280 | 253 | -27.7 [-34.2, -20.7] | 9.9 [7.6, 11.9] | 8.5 [6.0, 10.8] |
| B vs A | harmbench + strongreject + xstest_safe | 763 | 463 | 326 | -137.2 [-144.1, -130.5] | 29.6 [28.4, 30.8] | 27.7 [26.4, 29.0] |
| A_brief vs A | harmbench + strongreject + xstest_safe | 763 | 463 | 421 | -42.9 [-51.0, -34.5] | 9.3 [7.6, 10.8] | 8.5 [6.8, 10.1] |

## Start model safety

HarmBench classifier compliance of A in percent.

| Prompts | n | Compliance |
|---|---|---|
| first 100 harmbench prompts in file order | 100 | 73.0 [64.0, 81.0] |
| all harmful prompts | 513 | 60.4 [56.1, 64.5] |

## Truncation and broken answers

Cut means the answer hit 1024 tokens. Looping means some 10 word run appears at least 4 times. All samples are counted.

| Condition | Answers | Cut | Cut on length prompts | Empty | Looping |
|---|---|---|---|---|---|
| A | 1263 | 3.6% | 4.0% | 0 | 41 |
| B | 1263 | 0.9% | 1.7% | 0 | 11 |
| blend50 | 1263 | 1.5% | 1.7% | 0 | 15 |
| blend25 | 1263 | 2.3% | 3.0% | 0 | 20 |
| blend75 | 1263 | 1.0% | 2.0% | 0 | 6 |
| A_sample4 | 3052 | 2.5% | not run | 0 | 4 |
| A_brief | 763 | 0.7% | not run | 0 | 15 |
| rto_dpo | 1263 | 2.3% | 3.7% | 0 | 52 |
| rto_rdpo | 1263 | 2.0% | 3.3% | 0 | 37 |
| rto_simpo | 1263 | 0.0% | 0.0% | 0 | 0 |
| sft | 1263 | 2.5% | 2.3% | 0 | 69 |
| pr_mis_dpo | 1263 | 4.2% | 5.7% | 0 | 23 |
| pr_mis_rdpo | 1263 | 1.7% | 1.7% | 0 | 8 |
| pr_l3_dpo | 1263 | 1.3% | 1.0% | 0 | 0 |
| pr_l3_simpo | 1263 | 0.2% | 0.7% | 0 | 1 |

## Right files

Every condition checked against the pinned repo and revision or blend spec, and against the expected row count.

| Condition | Source | Revision | Matches pin | Rows |
|---|---|---|---|---|
| A | Salesforce/LLaMA-3-8B-SFR-Iterative-DPO-R | 157d91dc37 | yes | 1263 of 1263 |
| B | Salesforce/LLaMA-3-8B-SFR-Iterative-DPO-Concise-R | 46af8d2f90 | yes | 1263 of 1263 |
| blend50 | 0.5 A + 0.5 B | from A and B | yes | 1263 of 1263 |
| blend25 | 0.75 A + 0.25 B | from A and B | yes | 1263 of 1263 |
| blend75 | 0.25 A + 0.75 B | from A and B | yes | 1263 of 1263 |
| A_sample4 | Salesforce/LLaMA-3-8B-SFR-Iterative-DPO-R | 157d91dc37 | yes | 3052 of 3052 |
| A_brief | Salesforce/LLaMA-3-8B-SFR-Iterative-DPO-R | 157d91dc37 | yes | 763 of 763 |
| rto_dpo | RTO-RL/Llama3-8B-DPO | 5c7eecb273 | yes | 1263 of 1263 |
| rto_rdpo | RTO-RL/Llama3-8B-RDPO | bb678dcdbd | yes | 1263 of 1263 |
| rto_simpo | RTO-RL/Llama3-8B-SimPO | a9af4f2f40 | yes | 1263 of 1263 |
| sft | RLHFlow/LLaMA3-SFT | 1bedbf1899 | yes | 1263 of 1263 |
| pr_mis_dpo | princeton-nlp/Mistral-7B-Base-SFT-DPO | 8186136e31 | yes | 1263 of 1263 |
| pr_mis_rdpo | princeton-nlp/Mistral-7B-Base-SFT-RDPO | c6195f3f34 | yes | 1263 of 1263 |
| pr_l3_dpo | princeton-nlp/Llama-3-Instruct-8B-DPO-v0.2 | 0e0fc11603 | yes | 1263 of 1263 |
| pr_l3_simpo | princeton-nlp/Llama-3-Instruct-8B-SimPO-v0.2 | 2b18bdce3e | yes | 1263 of 1263 |
