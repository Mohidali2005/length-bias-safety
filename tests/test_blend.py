import json

import pytest
import torch
from safetensors.torch import load_file
from safetensors.torch import save_file

from lbs.generation.blend import blend

SHARDS = {"model-00001-of-00002.safetensors":["x"],"model-00002-of-00002.safetensors":["y","z"]}


def make_model(folder,seed,name="tiny"):
    """Write a tiny two shard model with random bfloat16 weights"""
    folder.mkdir()
    generator = torch.Generator().manual_seed(seed)
    weight_map = {}
    for shard,tensors in SHARDS.items():
        save_file({t:torch.randn(64,8,generator=generator).to(torch.bfloat16) for t in tensors},str(folder/shard),metadata={"format":"pt"})
        weight_map.update({t:shard for t in tensors})
    (folder/"model.safetensors.index.json").write_text(json.dumps({"metadata":{},"weight_map":weight_map}))
    (folder/"config.json").write_text(json.dumps({"_name_or_path":name,"hidden_size":8}))
    (folder/"tokenizer.json").write_text("{}")


def weights(folder):
    """Load every tensor of a tiny model into one dict"""
    return {k:v for shard in SHARDS for k,v in load_file(str(folder/shard)).items()}


def test_blend_mixes_weights_exactly(tmp_path):
    """Shares 0 and 1 give back A and B bit for bit and 0.25 matches the float32 formula"""
    make_model(tmp_path/"a",1,"a")
    make_model(tmp_path/"b",2,"b")
    a = weights(tmp_path/"a")
    b = weights(tmp_path/"b")
    for share in [0.0,0.25,1.0]:
        out = tmp_path/f"blend{share}"
        blend(tmp_path/"a",tmp_path/"b",share,out)
        mixed = weights(out)
        assert mixed.keys()==a.keys()
        for k in a:
            assert mixed[k].dtype==torch.bfloat16
            assert torch.equal(mixed[k],((1-share)*a[k].float()+share*b[k].float()).to(torch.bfloat16))
        if share==0.0:
            assert all(torch.equal(mixed[k],a[k]) for k in a)
        if share==1.0:
            assert all(torch.equal(mixed[k],b[k]) for k in a)
        assert (out/"tokenizer.json").exists() and (out/"model.safetensors.index.json").exists()
        assert json.loads((out/"config.json").read_text())["_name_or_path"]=="a"


def test_blend_refuses_models_that_do_not_match(tmp_path):
    """Different shard layouts or configs stop the blend"""
    make_model(tmp_path/"a",1)
    make_model(tmp_path/"b",2)
    (tmp_path/"b"/"config.json").write_text(json.dumps({"_name_or_path":"b","hidden_size":16}))
    with pytest.raises(AssertionError):
        blend(tmp_path/"a",tmp_path/"b",0.5,tmp_path/"out")
    make_model(tmp_path/"c",3)
    index = json.loads((tmp_path/"c"/"model.safetensors.index.json").read_text())
    index["weight_map"]["x"] = "model-00002-of-00002.safetensors"
    (tmp_path/"c"/"model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(AssertionError):
        blend(tmp_path/"a",tmp_path/"c",0.5,tmp_path/"out2")
