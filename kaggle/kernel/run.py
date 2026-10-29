import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from kaggle_secrets import UserSecretsClient

# smoke runs every code path on a few prompts and full runs the whole study
MODE = "smoke"
COMMIT = "0000000000000000000000000000000000000000"
REPO = "https://github.com/Mohidali2005/length-bias-safety.git"
INSTALL = ["vllm==0.32.0","peft==0.20.0","sentencepiece==0.2.2"]
CODE = Path("/tmp/code")
RESULTS = Path("/kaggle/working/results")
MODELS = Path("/tmp/lbs_models")
SMOKE_PROMPTS = Path("/tmp/smoke_prompts")
SMOKE_SIZE = 20
# a results folder of an earlier run attached as a dataset makes the full run continue from it
RESUME = Path("/kaggle/input/lbs-partial/results")
# sha256 of every prompt file built locally so the build here has to match it byte for byte
PROMPT_HASHES = {
    "harmbench":"45d124b42bbac957d8095e98a022698db7435f6727526d3906be9bb2afa8ff9c",
    "strongreject":"f42a82322fa345d80966738d94883b7493b1159c9a8a04b5e2bef34d0d04face",
    "xstest_safe":"021f75352559d0c079d035636be36075f9ee28daa7046562d999ec35374b4e50",
    "prefill":"97089598f8b4daa6dde5d122baccb8e0b8844d8a381f461250839e3be780d33f",
    "length":"4b0db501d41d38232273374ef9b6915d8b6069a626434a60eee4a24441a2f4a0",
}


def shell(command):
    """Run a shell command and return its output as text"""
    return subprocess.run(command,shell=True,capture_output=True,text=True,executable="/bin/bash").stdout.strip()


secrets = UserSecretsClient()
hf_token = secrets.get_secret("HF_TOKEN")
github_token = secrets.get_secret("GITHUB_TOKEN")

# passing the github token through git config variables so it never shows up in a command line or a file
auth = base64.b64encode(f"x-access-token:{github_token}".encode()).decode()
git_env = dict(os.environ,GIT_CONFIG_COUNT="1",GIT_CONFIG_KEY_0="http.extraHeader",GIT_CONFIG_VALUE_0=f"Authorization: Basic {auth}")
subprocess.run(["git","clone","--quiet",REPO,str(CODE)],env=git_env,check=True)
subprocess.run(["git","-C",str(CODE),"checkout","--quiet","--detach",COMMIT],check=True)
head = subprocess.run(["git","-C",str(CODE),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
assert head==COMMIT,f"checked out {head} instead of {COMMIT}"
print(f"code at commit {head}")

start = time.time()
subprocess.run([sys.executable,"-m","pip","install","--quiet",*INSTALL],check=True)
# the preinstalled torchaudio is built for another cuda version and breaks the transformers import while no audio is used
subprocess.run([sys.executable,"-m","pip","uninstall","--quiet","-y","torchaudio"],check=True)
print(f"installed packages in {(time.time()-start)/60:.1f} min")

env = dict(os.environ,PYTHONPATH=str(CODE/"src"),LBS_RESULTS=str(RESULTS),LBS_MODELS=str(MODELS),HF_TOKEN=hf_token,VLLM_LOGGING_LEVEL="WARNING")
subprocess.run([sys.executable,"-m","lbs.prompts"],env=env,check=True)
for name,expected in PROMPT_HASHES.items():
    assert hashlib.sha256((CODE/"data"/"prompts"/f"{name}.jsonl").read_bytes()).hexdigest()==expected,f"{name} prompts differ from the local build"
print("prompt files match the local build")

RESULTS.mkdir(parents=True,exist_ok=True)
(RESULTS/"environment.json").write_text(json.dumps({
    "mode":MODE,
    "commit":COMMIT,
    "started":time.strftime("%Y-%m-%d %H:%M:%S"),
    "gpus":shell("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
    "packages":shell("pip list 2>/dev/null | grep -i -E '^(torch|transformers|tokenizers|peft|vllm|sentencepiece|safetensors|huggingface.hub|accelerate|flashinfer.python|triton) '"),
    "python":sys.version,
},indent=1),encoding="utf-8")

if MODE=="full":
    resume = ["--resume-from",str(RESUME)] if RESUME.exists() else []
    subprocess.run([sys.executable,"-m","lbs.main","--kaggle",*resume],env=env,check=True)

if MODE=="smoke":
    SMOKE_PROMPTS.mkdir(parents=True,exist_ok=True)
    for f in (CODE/"data"/"prompts").glob("*.jsonl"):
        lines = f.read_text(encoding="utf-8").splitlines()[:SMOKE_SIZE]
        (SMOKE_PROMPTS/f.name).write_text("\n".join(lines)+"\n",encoding="utf-8")
    env["LBS_PROMPTS"] = str(SMOKE_PROMPTS)
    report = []

    def step(*args):
        """Run one module of the package and record its exit code time and the free disk"""
        began = time.time()
        code = subprocess.run([sys.executable,"-m",*[str(a) for a in args]],env=env).returncode
        free = shutil.disk_usage("/tmp").free/1e9
        report.append({"step":" ".join(str(a) for a in args),"exit_code":code,"minutes":round((time.time()-began)/60,2),"free_disk_gb":round(free)})
        (RESULTS/"smoke.json").write_text(json.dumps(report,indent=1),encoding="utf-8")
        print(f"smoke step {report[-1]}")

    step("lbs.fetch","A")
    step("lbs.generation.generate","A",MODELS/"A")
    step("lbs.fetch","B")
    step("lbs.generation.generate","B",MODELS/"B")
    step("lbs.fetch","blend50")
    step("lbs.generation.generate","blend50",MODELS/"blend50")
    step("lbs.generation.generate","A_sample4",MODELS/"A")
    step("lbs.generation.generate","A_brief",MODELS/"A")
    shutil.rmtree(MODELS,ignore_errors=True)
    for name in ["rto_dpo","pr_mis_dpo","pr_l3_dpo"]:
        step("lbs.fetch",name)
        step("lbs.generation.generate",name,MODELS/name)
        shutil.rmtree(MODELS,ignore_errors=True)
    step("lbs.fetch","harmbench_cls")
    step("lbs.scoring.harmbench")
    shutil.rmtree(MODELS,ignore_errors=True)
    step("lbs.fetch","wildguard")
    step("lbs.scoring.wildguard")
    shutil.rmtree(MODELS,ignore_errors=True)
    step("lbs.scoring.strmatch")
    step("lbs.fetch","gemma_2b","strongreject_adapter")
    step("lbs.scoring.strongreject")
    shutil.rmtree(MODELS,ignore_errors=True)
    failed = [r["step"] for r in report if r["exit_code"]!=0]
    print(f"smoke finished with {len(failed)} failed steps {failed}")
