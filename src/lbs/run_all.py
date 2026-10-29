import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from lbs import config
from lbs import records
from lbs.paths import GEN
from lbs.paths import MODELS
from lbs.paths import RESULTS
from lbs.paths import SCORES


def step(args):
    """Run one module of the package in its own process and say whether it worked"""
    return subprocess.run([sys.executable,"-m",*args]).returncode==0


def clear_models(keep):
    """Delete every model folder except the ones still needed"""
    for folder in MODELS.iterdir():
        if folder.name not in keep:
            shutil.rmtree(folder)


def run(resume_from=None,gen_hours=config.GEN_DEADLINE_H,hard_hours=config.HARD_DEADLINE_H):
    """Generate and score every condition in priority order while staying inside the time budget"""
    start = time.time()
    gen_deadline = start+gen_hours*3600
    hard_deadline = start+hard_hours*3600
    RESULTS.mkdir(parents=True,exist_ok=True)
    MODELS.mkdir(parents=True,exist_ok=True)

    if resume_from:
        # copying results of an earlier run without overwriting anything already here
        for src in Path(resume_from).rglob("*"):
            dst = RESULTS/src.relative_to(resume_from)
            if src.is_file() and not dst.exists():
                dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy(src,dst)

    gpus = 0
    if shutil.which("nvidia-smi"):
        gpus = len(subprocess.run(["nvidia-smi","-L"],capture_output=True,text=True).stdout.strip().splitlines())
        assert gpus>=config.TENSOR_PARALLEL,f"found {gpus} gpus but the models are split over {config.TENSOR_PARALLEL}"
    ram = os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES")/1e9 if hasattr(os,"sysconf") else 0
    print(f"gpus {gpus} | ram {ram:.0f} GB | free disk {shutil.disk_usage(MODELS).free/1e9:.0f} GB")

    manifest = {"started":time.strftime("%Y-%m-%d %H:%M:%S"),"conditions":{},"scorers":{}}
    names = list(config.CONDITIONS)
    for i,name in enumerate(names):
        condition = config.CONDITIONS[name]
        brief = condition.get("brief_from")
        if (GEN/f"{name}.done.json").exists():
            status = "done"
        elif time.time()>=gen_deadline:
            status = "skipped because the generation deadline passed"
        elif brief and not (GEN/f"{brief}.done.json").exists():
            status = f"skipped because {brief} has no answers"
        elif not step(["lbs.fetch",condition["model"]]):
            status = "failed while fetching the model"
        elif not step(["lbs.generation.generate",name,str(MODELS/condition["model"])]):
            status = "failed while generating"
        else:
            status = "done"
        manifest["conditions"][name] = status
        records.write_json(RESULTS/"manifest.json",manifest)

        # keeping only models that a later unfinished condition still needs so the disk does not fill up
        keep = set()
        for later in names[i+1:]:
            if not (GEN/f"{later}.done.json").exists():
                model = config.CONDITIONS[later]["model"]
                keep.add(model)
                keep.update(config.MODELS[model].get("blend",[])[:2])
        clear_models(keep)

    for scorer,settings in config.SCORERS.items():
        if not records.pending(scorer):
            status = "nothing to score"
        elif time.time()>=hard_deadline:
            status = "skipped because the hard deadline passed"
        elif not all(step(["lbs.fetch",model]) for model in settings["models"]):
            status = "failed while fetching the model"
        elif not step([f"lbs.scoring.{scorer}","--deadline",str(hard_deadline)]):
            status = "failed while scoring"
        elif records.pending(scorer):
            status = "stopped at the hard deadline"
        else:
            status = "done"
        manifest["scorers"][scorer] = status
        clear_models(set())
        records.write_json(RESULTS/"manifest.json",manifest)

    manifest["generated"] = [name for name in names if (GEN/f"{name}.done.json").exists()]
    manifest["scored"] = {scorer:[name for name in settings["conditions"] if (SCORES/scorer/f"{name}.done.json").exists()] for scorer,settings in config.SCORERS.items()}
    manifest["hours"] = round((time.time()-start)/3600,2)
    records.write_json(RESULTS/"manifest.json",manifest)
    print(f"finished in {manifest['hours']} h with {len(manifest['generated'])} of {len(names)} conditions generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run every gpu stage in one resumable pass")
    parser.add_argument("--resume-from")
    parser.add_argument("--gen-hours",type=float,default=config.GEN_DEADLINE_H)
    parser.add_argument("--hard-hours",type=float,default=config.HARD_DEADLINE_H)
    args = parser.parse_args()
    run(args.resume_from,args.gen_hours,args.hard_hours)
