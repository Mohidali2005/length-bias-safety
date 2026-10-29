import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from lbs import config
from lbs.generation.blend import blend
from lbs.paths import MODELS

# only safetensors weights so repos that also ship bin weights are not downloaded twice
PATTERNS = ["*.json","*.safetensors","tokenizer*"]


def fetch(name):
    """Put the files of one model into the models folder and return that folder"""
    target = MODELS/name
    if target.exists():
        return target
    MODELS.mkdir(parents=True,exist_ok=True)
    spec = config.MODELS[name]
    partial = MODELS/f"{name}.partial"
    shutil.rmtree(partial,ignore_errors=True)
    if "blend" in spec:
        a,b,share = spec["blend"]
        blend(fetch(a),fetch(b),share,partial)
    elif "LBS_LOCAL_MODELS" in os.environ:
        # the tests copy tiny local models instead of downloading the real ones
        shutil.copytree(Path(os.environ["LBS_LOCAL_MODELS"])/name,partial)
    else:
        snapshot_download(spec["repo"],revision=spec["revision"],local_dir=partial,allow_patterns=PATTERNS)
    # renaming only at the end so a half written folder is never taken for a finished one
    os.replace(partial,target)
    free = shutil.disk_usage(MODELS).free/1e9
    print(f"fetched {name} with {free:.0f} GB disk left")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download or blend models into the models folder")
    parser.add_argument("names",nargs="+")
    args = parser.parse_args()
    for name in args.names:
        fetch(name)
