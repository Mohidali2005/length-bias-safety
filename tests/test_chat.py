import pytest
from transformers import AutoTokenizer

from lbs import config
from lbs import records
from lbs.generation.chat import build_ids
from lbs.prompts import PREFILL

GENERATORS = [name for name,spec in config.MODELS.items() if "repo" in spec and "bos" in spec]


def load(name,tokenizer_dirs):
    """Load the real tokenizer of one model and the ids of its special tokens"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dirs[name])
    special = {i for i,t in tokenizer.added_tokens_decoder.items() if t.special}
    return tokenizer,special


@pytest.mark.parametrize("name",GENERATORS)
def test_bos_appears_exactly_as_in_training(name,tokenizer_dirs):
    """A prompt starts with one bos token for models trained with it and none for the others"""
    tokenizer,special = load(name,tokenizer_dirs)
    bos = config.MODELS[name]["bos"]
    ids = build_ids(tokenizer,"Hi there",None,bos)
    assert ids.count(tokenizer.bos_token_id)==int(bos)
    assert (ids[0]==tokenizer.bos_token_id)==bos


@pytest.mark.parametrize("name",GENERATORS)
def test_end_of_turn_token_stops_generation(name,tokenizer_dirs):
    """The token that closes an assistant turn in the template is one of the stop tokens"""
    tokenizer,special = load(name,tokenizer_dirs)
    stop = config.MODELS[name]["stop"]
    user = [{"role":"user","content":"Hi there"}]
    prefix = tokenizer.apply_chat_template(user,tokenize=False,add_generation_prompt=True)
    full = tokenizer.apply_chat_template(user+[{"role":"assistant","content":"OK"}],tokenize=False)
    assert full.startswith(prefix)
    closing = [i for i in tokenizer(full[len(prefix):],add_special_tokens=False).input_ids if i in special]
    assert closing[0] in stop
    assert set(tokenizer.convert_ids_to_tokens(stop))<={"<|end_of_text|>","<|eot_id|>","</s>"}


@pytest.mark.parametrize("name",GENERATORS)
def test_prefill_ends_the_prompt(name,tokenizer_dirs):
    """The prefilled opening is the last text of the prompt"""
    tokenizer,special = load(name,tokenizer_dirs)
    ids = build_ids(tokenizer,"Hi there",PREFILL,config.MODELS[name]["bos"])
    assert tokenizer.decode(ids).endswith(PREFILL)


@pytest.mark.parametrize("name",GENERATORS)
def test_every_prompt_fits_and_holds_no_special_tokens(name,tokenizer_dirs):
    """Every real prompt with the longest brief instruction leaves room for a full answer"""
    tokenizer,special = load(name,tokenizer_dirs)
    suffix = config.BRIEF.format(words=9999)
    for p in records.load_prompts(config.FULL):
        assert not special&set(tokenizer(p["prompt"],add_special_tokens=False).input_ids),p["id"]
        ids = build_ids(tokenizer,p["prompt"]+suffix,p["prefill"],config.MODELS[name]["bos"])
        assert len(ids)+config.MAX_NEW_TOKENS<=config.GEN_MAX_LEN,p["id"]
