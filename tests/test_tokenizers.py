import json

import pytest

from lbs import config
from lbs import records
from lbs.paths import TOKENIZERS
from lbs.scoring import tokens
from lbs.scoring.export_tokenizers import PROBE_ANSWERS
from lbs.scoring.export_tokenizers import PROBE_OUTPUTS
from lbs.scoring.export_tokenizers import digest
from lbs.scoring.templates import HARMBENCH
from lbs.scoring.templates import WILDGUARD


@pytest.mark.parametrize("name",["harmbench_cls","wildguard"])
def test_exported_tokenizers_reproduce_transformers_4(name):
    """The exported tokenizers give exactly the clips ids and decodes that transformers 4 gave"""
    reference = json.loads((TOKENIZERS/"reference.json").read_text(encoding="utf-8"))[name]
    tokenizer = tokens.load(name)
    prompts = [p["prompt"] for p in records.load_prompts(config.FULL)]
    clipped = [tokens.clip(tokenizer,a,config.HARMBENCH_CLIP) for a in PROBE_ANSWERS]
    if name=="harmbench_cls":
        texts = [HARMBENCH.format(behavior=p,generation=c) for p in prompts for c in clipped]
    else:
        texts = [WILDGUARD.format(prompt=p,response=a) for p in prompts for a in PROBE_ANSWERS]
    outputs = [tokenizer.decode(tokenizer.encode(o,add_special_tokens=False).ids,skip_special_tokens=True) for o in PROBE_OUTPUTS]
    assert len(texts)==reference["count"]
    assert digest(clipped)==reference["clipped"]
    assert digest([tokenizer.encode(t).ids for t in texts])==reference["inputs"]
    assert digest(outputs)==reference["outputs"]
