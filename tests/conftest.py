import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig
from peft import get_peft_model
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers import GemmaConfig
from transformers import GemmaForCausalLM

from lbs import config
from lbs.paths import PROMPTS
from lbs.paths import ROOT

STUBS = Path(__file__).parent/"stubs"
# letting test modules import code that imports vllm without a gpu
sys.path.insert(0,str(STUBS))

TOKENIZER_FILES = ["tokenizer*","special_tokens_map.json","added_tokens.json","config.json","generation_config.json"]


@pytest.fixture(scope="session")
def tokenizer_dirs():
    """Download the tokenizer and config files of every real model once at the pinned revisions"""
    dirs = {}
    for name,spec in config.MODELS.items():
        if "repo" in spec:
            folder = ROOT/"cache"/"test_tokenizers"/name
            snapshot_download(spec["repo"],revision=spec["revision"],local_dir=folder,allow_patterns=TOKENIZER_FILES)
            dirs[name] = folder
    return dirs


@pytest.fixture(scope="session")
def local_models(tmp_path_factory,tokenizer_dirs):
    """Build tiny stand ins for every model that keep the real tokenizers"""
    root = tmp_path_factory.mktemp("local_models")
    # copying file contents into fresh folders because onedrive marks folders read only and copytree would carry that over
    for name,folder in tokenizer_dirs.items():
        if name not in ["gemma_2b","strongreject_adapter"]:
            (root/name).mkdir()
            for f in folder.iterdir():
                if f.is_file():
                    shutil.copyfile(f,root/name/f.name)

    # giving models a and b two tiny shards so the real blend code has weights to mix
    shards = {"model-00001-of-00002.safetensors":["model.embed_tokens.weight"],"model-00002-of-00002.safetensors":["model.norm.weight","lm_head.weight"]}
    for name,seed in [("A",1),("B",2)]:
        generator = torch.Generator().manual_seed(seed)
        weight_map = {}
        for shard,tensors in shards.items():
            save_file({t:torch.randn(4,3,generator=generator).to(torch.bfloat16) for t in tensors},str(root/name/shard),metadata={"format":"pt"})
            weight_map.update({t:shard for t in tensors})
        (root/name/"model.safetensors.index.json").write_text(json.dumps({"metadata":{},"weight_map":weight_map}))

    # a tiny gemma with a random lora so the strongreject path runs end to end on cpu
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dirs["strongreject_adapter"])
    tiny = GemmaConfig(vocab_size=len(tokenizer),hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8)
    torch.manual_seed(0)
    base = GemmaForCausalLM(tiny)
    base.save_pretrained(root/"gemma_2b")
    lora = LoraConfig(r=8,lora_alpha=8,target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],init_lora_weights=False)
    get_peft_model(base,lora).save_pretrained(root/"strongreject_adapter")
    for f in tokenizer_dirs["strongreject_adapter"].iterdir():
        if f.is_file():
            shutil.copyfile(f,root/"strongreject_adapter"/f.name)
    return root


@pytest.fixture(scope="session")
def small_prompts(tmp_path_factory):
    """Copy the first five prompts of every set so pipeline runs stay fast"""
    folder = tmp_path_factory.mktemp("prompts")
    for name in config.FULL:
        lines = (PROMPTS/f"{name}.jsonl").read_text(encoding="utf-8").splitlines()[:5]
        (folder/f"{name}.jsonl").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return folder


@pytest.fixture(scope="session")
def make_env(local_models,small_prompts):
    """Return a function that builds the environment of one isolated pipeline run"""

    def env_for(folder,models=local_models):
        """Point results models prompts and the stub log into one folder"""
        env = dict(os.environ)
        env.update({
            "LBS_RESULTS":str(folder/"results"),
            "LBS_MODELS":str(folder/"models"),
            "LBS_LOCAL_MODELS":str(models),
            "LBS_PROMPTS":str(small_prompts),
            "LBS_STUB_LOG":str(folder/"stub_log.jsonl"),
            "PYTHONPATH":str(STUBS)+os.pathsep+env.get("PYTHONPATH",""),
        })
        return env

    return env_for


@pytest.fixture(scope="session")
def full_run(tmp_path_factory,make_env):
    """Run the whole pipeline once on cpu with the stub engine and the tiny models"""
    folder = tmp_path_factory.mktemp("full_run")
    result = subprocess.run([sys.executable,"-m","lbs.run_all"],env=make_env(folder),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return folder
