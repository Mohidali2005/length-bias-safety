import csv
import datetime
import hashlib
import json
import random

import requests

from lbs.paths import PROMPTS
from lbs.paths import RAW

SOURCES = {
    "harmbench":"https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "strongreject":"https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv",
    "xstest":"https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
    "alpaca_eval":"https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json",
}
EXPECTED = {"harmbench":200,"strongreject":313,"xstest_safe":250,"prefill":200,"length":300}
PREFILL = "Sure, here is"
SEED = 0


def build():
    """Download the benchmark files and write every prompt set as jsonl"""
    RAW.mkdir(parents=True,exist_ok=True)
    PROMPTS.mkdir(parents=True,exist_ok=True)

    files = {}
    for name,url in SOURCES.items():
        path = RAW/url.rsplit("/",1)[1]
        if not path.exists():
            response = requests.get(url,timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        files[name] = path

    with open(files["harmbench"],encoding="utf-8") as f:
        harmbench = [r for r in csv.DictReader(f) if r["FunctionalCategory"]=="standard"]
    with open(files["strongreject"],encoding="utf-8") as f:
        strongreject = list(csv.DictReader(f))
    with open(files["xstest"],encoding="utf-8") as f:
        xstest = list(csv.DictReader(f))
    alpaca = json.loads(files["alpaca_eval"].read_text(encoding="utf-8"))

    sets = {}
    sets["harmbench"] = [{"id":"hb_"+r["BehaviorID"],"category":r["SemanticCategory"],"prompt":r["Behavior"],"prefill":None} for r in harmbench]
    sets["strongreject"] = [{"id":f"sr_{i:03d}","category":r["category"],"prompt":r["forbidden_prompt"],"prefill":None} for i,r in enumerate(strongreject)]
    sets["xstest_safe"] = [{"id":f"xs_{r['id']}","category":r["type"],"prompt":r["prompt"],"prefill":None} for r in xstest if r["label"]=="safe"]
    # reusing the same harmbench behaviors so prefill compliance can be paired with plain compliance
    sets["prefill"] = [{**r,"id":"pf_"+r["id"][3:],"prefill":PREFILL} for r in sets["harmbench"]]
    # sorting the sampled indices so the length set keeps the original alpaca order
    indices = sorted(random.Random(SEED).sample(range(len(alpaca)),EXPECTED["length"]))
    sets["length"] = [{"id":f"al_{i:03d}","category":alpaca[i]["dataset"],"prompt":alpaca[i]["instruction"],"prefill":None} for i in indices]

    for name,records in sets.items():
        ids = [r["id"] for r in records]
        assert len(records)==EXPECTED[name],f"{name} has {len(records)} prompts"
        assert len(set(ids))==len(ids),f"{name} has duplicate ids"
        assert all(r["prompt"].strip() for r in records),f"{name} has an empty prompt"
        with open(PROMPTS/f"{name}.jsonl","w",encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({"set":name,**r},ensure_ascii=False)+"\n")
        print(f"{name}: {len(records)} prompts")

    manifest = {
        "built":datetime.date.today().isoformat(),
        "seed":SEED,
        "sources":{name:{"url":SOURCES[name],"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in files.items()},
        "counts":{name:len(records) for name,records in sets.items()},
    }
    (PROMPTS/"manifest.json").write_text(json.dumps(manifest,indent=1),encoding="utf-8")


if __name__ == "__main__":
    build()
