import argparse
import collections
import statistics
import time

import vllm
from transformers import AutoTokenizer

from lbs import config
from lbs import records
from lbs.generation.chat import build_ids
from lbs.paths import GEN

parser = argparse.ArgumentParser(description="generate answers for one condition")
parser.add_argument("condition")
parser.add_argument("model_dir")
args = parser.parse_args()

condition = config.CONDITIONS[args.condition]
model = config.MODELS[condition["model"]]
prompts = records.load_prompts(condition["sets"])
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

suffix = ""
words = None
if "brief_from" in condition:
    reference = [r for r in records.read_jsonl(GEN/f"{condition['brief_from']}.jsonl") if r["set"] in condition["sets"]]
    words = round(statistics.mean(len(r["response"].split()) for r in reference))
    suffix = config.BRIEF.format(words=words)

prompt_ids = [build_ids(tokenizer,p["prompt"]+suffix,p["prefill"],model["bos"]) for p in prompts]
longest = max(len(ids) for ids in prompt_ids)
assert longest+config.MAX_NEW_TOKENS<=config.GEN_MAX_LEN,f"a prompt of {longest} tokens leaves no room for the answer"
# repeating each prompt instead of asking for n samples so every sample is drawn on its own
jobs = [(p,ids,s) for p,ids in zip(prompts,prompt_ids) for s in range(condition["samples"])]

llm = vllm.LLM(model=args.model_dir,dtype=config.DTYPE,tensor_parallel_size=config.TENSOR_PARALLEL,max_model_len=config.GEN_MAX_LEN,gpu_memory_utilization=config.GPU_MEMORY,seed=config.SEED)
params = vllm.SamplingParams(n=1,temperature=condition["temperature"],top_p=1.0,repetition_penalty=1.0,max_tokens=config.MAX_NEW_TOKENS,stop_token_ids=model["stop"])
start = time.perf_counter()
outputs = llm.generate([{"prompt_token_ids":ids} for p,ids,s in jobs],params)
seconds = time.perf_counter()-start

rows = []
for (p,ids,s),out in zip(jobs,outputs):
    answer = out.outputs[0]
    rows.append({
        "condition":args.condition,
        "set":p["set"],
        "id":p["id"],
        "sample":s,
        "prompt":p["prompt"],
        "message":p["prompt"]+suffix,
        "prefill":p["prefill"],
        "response":answer.text,
        "n_tokens":len(answer.token_ids),
        "finish_reason":answer.finish_reason,
    })
records.write_jsonl(GEN/f"{args.condition}.jsonl",rows)

finish = collections.Counter(r["finish_reason"] for r in rows)
tokens = sum(r["n_tokens"] for r in rows)
records.write_json(GEN/f"{args.condition}.done.json",{
    "condition":args.condition,
    "model":condition["model"],
    "repo":model.get("repo"),
    "revision":model.get("revision"),
    "blend":model.get("blend"),
    "rows":len(rows),
    "seconds":round(seconds),
    "tokens_per_second":round(tokens/seconds),
    "finish_reasons":dict(finish),
    "brief_words":words,
    "longest_prompt":longest,
    "vllm":vllm.__version__,
})
print(f"{args.condition}: {len(rows)} answers in {seconds/60:.1f} min with {finish['length']} cut at the token limit")
