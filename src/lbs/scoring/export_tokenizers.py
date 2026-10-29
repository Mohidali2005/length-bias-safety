import hashlib
import json
import tempfile

import transformers
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from lbs import config
from lbs import records
from lbs.paths import TOKENIZERS
from lbs.scoring.templates import HARMBENCH
from lbs.scoring.templates import WILDGUARD

# answers chosen to hit leading spaces unicode code blocks byte fallback and truncation
PROBE_ANSWERS = [
    "I cannot help with that request.",
    "Sure, here is a detailed answer:\n\n1. First step\n2. Second step\n\n**Note:** be careful — it’s “important”.",
    "```python\nimport os\nprint(os.getcwd())\n```\n\nThat prints the folder.",
    "Je ne peux pas aider. 我不能帮忙。 \U0001F642  double  spaces\tand tabs\n\n\n",
    " leading space and trailing space ",
    "word "*900,
]
PROBE_OUTPUTS = ["Yes","No"," Yes","yes","Harmful request: yes\nResponse refusal: no\nHarmful response: no"]


def digest(data):
    """Hash any json serialisable value"""
    return hashlib.sha256(json.dumps(data,ensure_ascii=False).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    # the classifiers were trained and validated with transformers 4 fast tokenizers so their behaviour is the reference
    assert transformers.__version__.startswith("4."),"run this export with transformers 4"
    prompts = [p["prompt"] for p in records.load_prompts(config.FULL)]
    reference = {"transformers":transformers.__version__}
    TOKENIZERS.mkdir(parents=True,exist_ok=True)
    for name in ["harmbench_cls","wildguard"]:
        spec = config.MODELS[name]
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download(spec["repo"],revision=spec["revision"],local_dir=tmp,allow_patterns=["tokenizer*","special_tokens_map.json","added_tokens.json"])
            tokenizer = AutoTokenizer.from_pretrained(tmp)
            assert not tokenizer.clean_up_tokenization_spaces,"decode would add a cleanup step the rust tokenizer does not do"
            tokenizer.backend_tokenizer.save(str(TOKENIZERS/f"{name}.json"))
        tokenizer.truncation_side = "right"
        clipped = [tokenizer.decode(tokenizer.encode(a,max_length=config.HARMBENCH_CLIP,truncation=True),skip_special_tokens=True) for a in PROBE_ANSWERS]
        if name=="harmbench_cls":
            texts = [HARMBENCH.format(behavior=p,generation=c) for p in prompts for c in clipped]
        else:
            texts = [WILDGUARD.format(prompt=p,response=a) for p in prompts for a in PROBE_ANSWERS]
        outputs = [tokenizer.decode(tokenizer(o,add_special_tokens=False).input_ids,skip_special_tokens=True) for o in PROBE_OUTPUTS]
        reference[name] = {"clipped":digest(clipped),"inputs":digest([tokenizer(t).input_ids for t in texts]),"outputs":digest(outputs),"count":len(texts)}
        print(f"{name}: saved tokenizer and reference over {len(texts)} inputs")
    (TOKENIZERS/"reference.json").write_text(json.dumps(reference,indent=1),encoding="utf-8",newline="\n")
