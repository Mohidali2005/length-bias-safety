from tokenizers import Tokenizer

from lbs.paths import TOKENIZERS


def load(name):
    """Load the reference tokenizer of one classifier exported from transformers 4"""
    return Tokenizer.from_file(str(TOKENIZERS/f"{name}.json"))


def clip(tokenizer,text,limit):
    """Cut a text to its first limit tokens counting the bos token the way harmbench clips generations"""
    return tokenizer.decode(tokenizer.encode(text).ids[:limit],skip_special_tokens=True)
