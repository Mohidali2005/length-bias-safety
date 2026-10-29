import json
import os
import subprocess
import sys
import time
from pathlib import Path

VLLM = "vllm==0.32.0"
# a small model with the llama architecture so the probe goes through the same vllm code as the real models
TINY = "HuggingFaceTB/SmolLM2-360M-Instruct"
PACKAGES = "torch|transformers|tokenizers|peft|vllm|sentencepiece|safetensors|huggingface.hub|accelerate|flashinfer.python|xformers|triton"
OUT = Path("/kaggle/working/probe.json")

TEST = '''
import time
import vllm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("MODEL")
questions = ["How do I boil an egg?","Explain photosynthesis to a child.","Write a short poem about rain.","What is the capital of France?"]*16
texts = [tokenizer.apply_chat_template([{"role":"user","content":q}],tokenize=False,add_generation_prompt=True) for q in questions]
ids = [tokenizer(t,add_special_tokens=False).input_ids for t in texts]
llm = vllm.LLM(model="MODEL",dtype="float16",tensor_parallel_size=2,max_model_len=2048,gpu_memory_utilization=0.9,seed=0)
params = vllm.SamplingParams(n=1,temperature=0.0,top_p=1.0,repetition_penalty=1.0,max_tokens=256,stop_token_ids=[tokenizer.eos_token_id])
start = time.perf_counter()
outputs = llm.generate([{"prompt_token_ids":i} for i in ids],params)
seconds = time.perf_counter()-start
tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print("PROBE tokens per second",round(tokens/seconds),"over",len(outputs),"prompts")
print("PROBE finish reasons",sorted({o.outputs[0].finish_reason for o in outputs}))
print("PROBE sample answer",repr(outputs[0].outputs[0].text[:300]))
text_outputs = llm.generate(["The capital of France is"],vllm.SamplingParams(temperature=0.0,max_tokens=1))
print("PROBE string prompt answer",repr(text_outputs[0].outputs[0].text))
'''.replace("MODEL",TINY)


def shell(command):
    """Run a shell command and return its combined output as text"""
    return subprocess.run(command,shell=True,capture_output=True,text=True,executable="/bin/bash").stdout.strip()


def save(facts):
    """Write what is known so far so a crash later still leaves the earlier facts"""
    OUT.write_text(json.dumps(facts,indent=1))


facts = {"python":sys.version,"cpus":os.cpu_count()}
facts["gpus"] = shell("nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>&1")
facts["nvidia_smi"] = shell("nvidia-smi 2>&1")
facts["disk"] = shell("df -h / /tmp /kaggle/working 2>&1")
facts["ram"] = shell("free -g 2>&1")
facts["packages_before"] = shell(f"pip list 2>/dev/null | grep -i -E '^({PACKAGES}) '")
save(facts)
print("probe: machine facts saved")

start = time.time()
facts["install_tail"] = shell(f"pip install {VLLM} peft sentencepiece 2>&1 | tail -25")
# the preinstalled torchaudio is built for another cuda version and breaks the transformers import while no audio is used
facts["uninstall_torchaudio"] = shell("pip uninstall -y torchaudio 2>&1 | tail -2")
facts["install_minutes"] = round((time.time()-start)/60,1)
facts["packages_after"] = shell(f"pip list 2>/dev/null | grep -i -E '^({PACKAGES}) '")
facts["torch_cuda"] = shell("python -c 'import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.device_count(),torch.cuda.get_device_capability(0))' 2>&1 | tail -3")
facts["vllm_tokenizer_modes"] = shell("grep -n -A8 '^TokenizerMode = Literal' $(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')/config/model.py 2>&1")
facts["imports"] = shell("python -c 'import transformers,vllm,peft;print(transformers.__version__,vllm.__version__,peft.__version__)' 2>&1 | tail -3")
save(facts)
print(f"probe: install took {facts['install_minutes']} min")

Path("/tmp/vllm_test.py").write_text(TEST)
start = time.time()
shell("python /tmp/vllm_test.py > /tmp/vllm_test.log 2>&1")
facts["vllm_test"] = shell("grep -E 'PROBE|Error|error|backend|Backend|Traceback' /tmp/vllm_test.log | tail -40")
facts["vllm_test_tail"] = shell("tail -60 /tmp/vllm_test.log")
facts["vllm_test_minutes"] = round((time.time()-start)/60,1)
save(facts)
print("probe: vllm test finished")

# timing one five gigabyte shard to estimate download time for the full run
start = time.time()
shell("python -c \"from huggingface_hub import hf_hub_download;hf_hub_download('Salesforce/LLaMA-3-8B-SFR-Iterative-DPO-R','model-00001-of-00004.safetensors',revision='157d91dc372d87d6deb3de241f9b9bc0d2e9026e',local_dir='/tmp/probe_shard')\" 2>&1")
facts["download_mb_per_second"] = round(4976.7/(time.time()-start))
shell("rm -rf /tmp/probe_shard")
facts["disk_after"] = shell("df -h / /tmp /kaggle/working 2>&1")
save(facts)
print(f"probe: done with download speed {facts['download_mb_per_second']} MB per second")
