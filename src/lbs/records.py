import json
import os

from lbs import config
from lbs.paths import GEN
from lbs.paths import PROMPTS
from lbs.paths import SCORES


def read_jsonl(path):
    """Read a jsonl file into a list of dicts"""
    with open(path,encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path,rows):
    """Write rows to a jsonl file through a temporary file so a crash never leaves half a file"""
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    with open(tmp,"w",encoding="utf-8",newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    os.replace(tmp,path)


def write_json(path,data):
    """Write one json file through a temporary file so a crash never leaves half a file"""
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(data,indent=1,ensure_ascii=False),encoding="utf-8",newline="\n")
    os.replace(tmp,path)


def load_prompts(sets):
    """Load the prompt records of the given sets in a fixed order"""
    return [row for name in sets for row in read_jsonl(PROMPTS/f"{name}.jsonl")]


def pending(scorer):
    """List the conditions that have answers but no scores yet from one scorer"""
    names = config.SCORERS[scorer]["conditions"]
    return [name for name in names if (GEN/f"{name}.done.json").exists() and not (SCORES/scorer/f"{name}.done.json").exists()]
