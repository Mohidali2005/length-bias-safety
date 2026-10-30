import csv
import random

import numpy as np

from lbs import config
from lbs import records
from lbs.paths import ANALYSIS
from lbs.paths import RESULTS
from lbs.paths import RUN

# the sheet holds harmful text so it stays in the ignored results folder
SHEET = RESULTS/"hand_labels"/"sheet.csv"
KEY = RESULTS/"hand_labels"/"key.csv"
PER_CELL = 25
REFUSAL = ["full_compliance","partial_refusal","full_refusal"]


def kappa(x,y):
    """Compute the agreement and cohen kappa of two yes or no arrays"""
    agree = (x==y).mean()
    chance = x.mean()*y.mean()+(1-x.mean())*(1-y.mean())
    return agree,(agree-chance)/(1-chance) if chance<1 else float("nan")


def make_sheet():
    """Sample answers of A and B for blind labelling and write the sheet and its separate key"""
    rng = random.Random(config.SEED)
    answers = {c:{(r["set"],r["id"]):r for r in records.read_jsonl(RUN/"gen"/f"{c}.jsonl")} for c in ["A","B"]}
    items = []
    for sets in [["harmbench","strongreject"],["xstest_safe"]]:
        keys = sorted(k for k in answers["A"] if k[0] in sets)
        # drawing different prompts for A and B so no pair of answers to one prompt gives the model away
        picked = rng.sample(keys,2*PER_CELL)
        items += [("A",k) for k in picked[:PER_CELL]]+[("B",k) for k in picked[PER_CELL:]]
    rng.shuffle(items)

    SHEET.parent.mkdir(parents=True,exist_ok=True)
    with open(SHEET,"w",encoding="utf-8-sig",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item","prompt","response","comply","refusal","note"])
        for n,(c,k) in enumerate(items,1):
            writer.writerow([n,answers[c][k]["prompt"],answers[c][k]["response"],"","",""])
    with open(KEY,"w",encoding="utf-8",newline="") as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerow(["item","condition","set","id"])
        for n,(c,k) in enumerate(items,1):
            writer.writerow([n,c,k[0],k[1]])
    print(f"hand label sheet written with {len(items)} answers")


def agreement():
    """Compare the filled hand labels with every scorer and write the agreement tables"""
    with open(SHEET,encoding="utf-8-sig",newline="") as f:
        sheet = {r["item"]:r for r in csv.DictReader(f)}
    with open(KEY,encoding="utf-8",newline="") as f:
        key = {r["item"]:r for r in csv.DictReader(f)}
    for n,r in sheet.items():
        assert r["refusal"] in REFUSAL,f"item {n} has refusal label {r['refusal']!r}"
        assert key[n]["set"]=="xstest_safe" or r["comply"] in ["0","1"],f"item {n} has comply label {r['comply']!r}"

    scores = {}
    for scorer,field in [("harmbench","comply"),("wildguard","harmful_response"),("wildguard","refusal"),("strmatch","refusal")]:
        for c in ["A","B"]:
            for r in records.read_jsonl(RUN/"scores"/scorer/f"{c}.jsonl"):
                scores[(scorer,field,c,r["set"],r["id"])] = r[field]

    labels = [{"item":n,"condition":key[n]["condition"],"set":key[n]["set"],"id":key[n]["id"],"comply":r["comply"],"refusal":r["refusal"]}
              for n,r in sheet.items()]
    with open(ANALYSIS/"hand_labels.csv","w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(labels[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(labels)

    lines = ["# Classifier agreement with hand labels","",
             "Refusal counts both full and partial refusals. Compliance is labelled only on harmful prompts. "
             "A large gap between A and B means the scorer treats long and short answers differently.","",
             "| Scorer | Compared with | Prompts | Condition | n | Agreement | Cohen kappa |","|---|---|---|---|---|---|---|"]
    for scorer,field,target,sets in [("harmbench","comply","comply",["harmbench","strongreject"]),
                                     ("wildguard","harmful_response","comply",["harmbench","strongreject"]),
                                     ("wildguard","refusal","refusal",["harmbench","strongreject"]),
                                     ("wildguard","refusal","refusal",["xstest_safe"]),
                                     ("strmatch","refusal","refusal",["harmbench","strongreject"]),
                                     ("strmatch","refusal","refusal",["xstest_safe"])]:
        for c in ["A","B","both"]:
            chosen = [l for l in labels if l["set"] in sets and c in [l["condition"],"both"]]
            human = np.array([int(l["comply"]) if target=="comply" else int(l["refusal"]!="full_compliance") for l in chosen])
            machine = np.array([scores[(scorer,field,l["condition"],l["set"],l["id"])] for l in chosen])
            agree,k = kappa(human,machine)
            lines.append(f"| {scorer} {field} | hand {target} | {' + '.join(sets)} | {c} | {len(chosen)} | {100*agree:.0f} | {k:.2f} |")
    lines.append("")
    (ANALYSIS/"stats").mkdir(parents=True,exist_ok=True)
    (ANALYSIS/"stats"/"classifier_agreement.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
    print(f"agreement written for {len(labels)} hand labels")


def run():
    """Write the sheet the first time and the agreement once every answer is labelled"""
    if not SHEET.exists():
        make_sheet()
        return
    with open(SHEET,encoding="utf-8-sig",newline="") as f:
        filled = [r["refusal"]!="" for r in csv.DictReader(f)]
    if all(filled):
        agreement()
    else:
        print(f"hand label sheet has {sum(filled)} of {len(filled)} answers labelled")


if __name__ == "__main__":
    run()
