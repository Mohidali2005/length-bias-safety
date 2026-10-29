import json
import re

import requests

from lbs import config
from lbs.scoring.templates import HARMBENCH
from lbs.scoring.templates import STRONGREJECT
from lbs.scoring.templates import WILDGUARD


def fetch(url):
    """Download one upstream text file"""
    response = requests.get(url,timeout=60)
    response.raise_for_status()
    return response.text


def test_harmbench_template_matches_the_classifier_card():
    """The harmbench prompt equals the template on the classifier card at the pinned revision"""
    revision = config.MODELS["harmbench_cls"]["revision"]
    card = fetch(f"https://huggingface.co/cais/HarmBench-Mistral-7b-val-cls/raw/{revision}/README.md")
    assert HARMBENCH==re.search(r"\"prompt\": '''(.*?)''',",card,re.S).group(1)


def test_wildguard_template_matches_the_official_package():
    """The wildguard prompt equals the input format of the official package"""
    source = fetch("https://raw.githubusercontent.com/allenai/wildguard/main/wildguard/utils.py")
    assert WILDGUARD==re.search(r'WILDGUARD_INPUT_FORMAT = """(.*?)"""',source,re.S).group(1)


def test_strongreject_template_matches_the_official_package():
    """The strongreject prompt equals the fine tuned judge template of the official package"""
    source = fetch("https://raw.githubusercontent.com/dsbowen/strong_reject/main/strong_reject/eval_files/judge_templates.json")
    assert STRONGREJECT==json.loads(source)["strongreject_finetuned"]
