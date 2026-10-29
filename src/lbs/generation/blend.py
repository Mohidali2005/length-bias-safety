import json
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


def blend(a_dir,b_dir,share,out_dir):
    """Write a model whose weights mix model A and model B with the given share of B"""
    index_a = json.loads((a_dir/"model.safetensors.index.json").read_text())
    index_b = json.loads((b_dir/"model.safetensors.index.json").read_text())
    assert index_a["weight_map"]==index_b["weight_map"],"A and B are split into shards differently"
    config_a = json.loads((a_dir/"config.json").read_text())
    config_b = json.loads((b_dir/"config.json").read_text())
    config_a.pop("_name_or_path",None)
    config_b.pop("_name_or_path",None)
    assert config_a==config_b,"A and B have different model configs"

    out_dir.mkdir(parents=True)
    for shard in sorted(set(index_a["weight_map"].values())):
        mixed = {}
        # reading one tensor at a time so only one mixed shard sits in memory
        with safe_open(str(a_dir/shard),framework="pt") as fa,safe_open(str(b_dir/shard),framework="pt") as fb:
            assert set(fa.keys())==set(fb.keys()),f"{shard} holds different tensors in A and B"
            for name in fa.keys():
                a = fa.get_tensor(name)
                b = fb.get_tensor(name)
                assert a.shape==b.shape and a.dtype==b.dtype,f"{name} differs in shape or dtype"
                # mixing in float32 so rounding back to the stored dtype happens only once
                mixed[name] = ((1-share)*a.float()+share*b.float()).to(a.dtype)
        save_file(mixed,str(out_dir/shard),metadata={"format":"pt"})

    for f in a_dir.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy(f,out_dir/f.name)
