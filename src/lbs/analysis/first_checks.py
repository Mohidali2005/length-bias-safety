import json
import random
from collections import Counter

import numpy as np

from lbs import config
from lbs import records
from lbs.paths import ANALYSIS
from lbs.paths import PROMPTS
from lbs.paths import RUN

RESAMPLES = 10000
HARMFUL = ["harmbench","strongreject"]
# each pair lists the shorter model first and its long reference second
PAIRS = [("B","A"),("rto_rdpo","rto_dpo"),("rto_simpo","rto_dpo"),("pr_mis_rdpo","pr_mis_dpo"),("pr_l3_simpo","pr_l3_dpo")]
# a ten word run repeated this often only shows up when the model is stuck in a loop
LOOP_WINDOW = 10
LOOP_REPEATS = 4


def answers(name):
    """Load the first sample of every answer of one condition keyed by set and prompt id"""
    return {(r["set"],r["id"]):r for r in records.read_jsonl(RUN/"gen"/f"{name}.jsonl") if r["sample"]==0}


def resample(n):
    """Draw the same bootstrap resamples of prompt indices for every statistic"""
    return np.random.default_rng(config.SEED).integers(0,n,(RESAMPLES,n))


def interval(values):
    """Format a point estimate with the middle 95 percent of its bootstrap values"""
    lo,hi = np.percentile(values[1],[2.5,97.5])
    return f"{values[0]:.1f} [{lo:.1f}, {hi:.1f}]"


def gap(short,long,sets):
    """Measure how much shorter one condition answers than another on the same prompts as a markdown row"""
    a,b = answers(short),answers(long)
    keys = [k for k in b if k[0] in sets]
    idx = resample(len(keys))
    cells = []
    for field in ["words","tokens"]:
        if field=="words":
            s = np.array([len(a[k]["response"].split()) for k in keys])
            l = np.array([len(b[k]["response"].split()) for k in keys])
        else:
            s = np.array([a[k]["n_tokens"] for k in keys])
            l = np.array([b[k]["n_tokens"] for k in keys])
        shorter = 100*(1-s.mean()/l.mean()),100*(1-s[idx].mean(1)/l[idx].mean(1))
        if field=="words":
            cells += [f"{l.mean():.0f}",f"{s.mean():.0f}",interval(((s-l).mean(),(s-l)[idx].mean(1)))]
        cells.append(interval(shorter))
    return f"| {short} vs {long} | {' + '.join(sets)} | {len(keys)} | "+" | ".join(cells)+" |"


def write():
    """Write the descriptive checks of the full run to one markdown file"""
    environment = json.loads((RUN/"environment.json").read_text(encoding="utf-8"))
    lines = ["# First checks",""]
    lines.append(f"Run `{RUN.parent.name}`, code commit `{environment['commit'][:7]}`, started {environment['started']} UTC. "
                 f"Intervals are paired bootstrap 95% intervals over prompts with {RESAMPLES:,} resamples.")
    lines.append("")

    lines += ["## Length gaps","",
              "Words are whitespace words, tokens are each model's own tokens. A negative difference means the first model is shorter.","",
              "| Pair | Prompts | n | Long words | Short words | Word difference | % shorter (words) | % shorter (tokens) |",
              "|---|---|---|---|---|---|---|---|"]
    for short,long in PAIRS:
        lines.append(gap(short,long,["length"]))
    lines.append(gap("B","A",config.SAFETY))
    lines.append(gap("A_brief","A",config.SAFETY))
    lines.append("")

    scores = {(r["set"],r["id"]):r["comply"] for r in records.read_jsonl(RUN/"scores"/"harmbench"/"A.jsonl")}
    first = [r["id"] for r in records.read_jsonl(PROMPTS/"harmbench.jsonl")][:100]
    lines += ["## Start model safety","","HarmBench classifier compliance of A in percent.","","| Prompts | n | Compliance |","|---|---|---|"]
    for label,keys in [("first 100 harmbench prompts in file order",[("harmbench",i) for i in first]),
                       ("all harmful prompts",[k for k in scores if k[0] in HARMFUL])]:
        x = 100*np.array([scores[k]==1 for k in keys])
        lines.append(f"| {label} | {len(keys)} | {interval((x.mean(),x[resample(len(x))].mean(1)))} |")
    lines.append("")

    lines += ["## Truncation and broken answers","",
              f"Cut means the answer hit {config.MAX_NEW_TOKENS} tokens. Looping means some {LOOP_WINDOW} word run appears at least {LOOP_REPEATS} times. "
              "All samples are counted.","",
              "| Condition | Answers | Cut | Cut on length prompts | Empty | Looping |","|---|---|---|---|---|---|"]
    for name in config.CONDITIONS:
        rows = records.read_jsonl(RUN/"gen"/f"{name}.jsonl")
        cut = [r["finish_reason"]=="length" for r in rows]
        cut_length = [r["finish_reason"]=="length" for r in rows if r["set"]=="length"]
        empty = sum(not r["response"].strip() for r in rows)
        looping = 0
        for r in rows:
            w = r["response"].split()
            runs = Counter(tuple(w[i:i+LOOP_WINDOW]) for i in range(len(w)-LOOP_WINDOW+1))
            looping += bool(runs) and max(runs.values())>=LOOP_REPEATS
        share_length = f"{100*np.mean(cut_length):.1f}%" if cut_length else "not run"
        lines.append(f"| {name} | {len(rows)} | {100*np.mean(cut):.1f}% | {share_length} | {empty} | {looping} |")
    lines.append("")

    lines += ["## Right files","",
              "Every condition checked against the pinned repo and revision or blend spec, and against the expected row count.","",
              "| Condition | Source | Revision | Matches pin | Rows |","|---|---|---|---|---|"]
    for name,spec in config.CONDITIONS.items():
        done = json.loads((RUN/"gen"/f"{name}.done.json").read_text(encoding="utf-8"))
        model = config.MODELS[spec["model"]]
        expected = len(records.load_prompts(spec["sets"]))*spec["samples"]
        if "blend" in model:
            source,revision,pinned = f"{1-model['blend'][2]:g} A + {model['blend'][2]:g} B","from A and B",done["blend"]==model["blend"]
        else:
            source,revision,pinned = done["repo"],done["revision"][:10],(done["repo"],done["revision"])==(model["repo"],model["revision"])
        lines.append(f"| {name} | {source} | {revision} | {'yes' if pinned and done['model']==spec['model'] else 'NO'} | {done['rows']} of {expected} |")
    lines.append("")

    # sampling only alpaca prompts so the file holds no harmful text
    blend = answers("blend50")
    picked = random.Random(config.SEED).sample(sorted(k for k in blend if k[0]=="length"),20)
    sample = ["# blend50 answers for the fluency read",""]
    for k in picked:
        sample += [f"## {k[1]} ({blend[k]['finish_reason']}, {len(blend[k]['response'].split())} words)","",
                   "**Prompt:** "+blend[k]["prompt"],"",blend[k]["response"],""]

    (ANALYSIS/"transcripts").mkdir(parents=True,exist_ok=True)
    (ANALYSIS/"first_checks.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
    (ANALYSIS/"transcripts"/"blend50_fluency.md").write_text("\n".join(sample),encoding="utf-8",newline="\n")
    print(f"first checks written for {len(config.CONDITIONS)} conditions and 20 blend50 answers set aside for reading")


if __name__ == "__main__":
    write()
