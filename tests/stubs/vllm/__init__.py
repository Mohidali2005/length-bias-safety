import atexit
import json
import os

from tokenizers import Tokenizer

from lbs.paths import TOKENIZERS

__version__ = "stub"


def log(event):
    """Append one event to the stub log so tests can check what the pipeline asked for"""
    if "LBS_STUB_LOG" in os.environ:
        with open(os.environ["LBS_STUB_LOG"],"a",encoding="utf-8") as f:
            f.write(json.dumps(event)+"\n")


class SamplingParams:
    """Hold sampling settings under the same names as the real vllm class"""

    def __init__(self,n=1,temperature=1.0,top_p=1.0,repetition_penalty=1.0,max_tokens=16,stop_token_ids=None):
        """Store the settings"""
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.stop_token_ids = stop_token_ids


class Completion:
    """One generated answer"""

    def __init__(self,text,token_ids,finish_reason):
        """Store the answer"""
        self.text = text
        self.token_ids = token_ids
        self.finish_reason = finish_reason


class Output:
    """All answers to one prompt"""

    def __init__(self,outputs):
        """Store the answers"""
        self.outputs = outputs


class LLM:
    """Pretend to be a vllm engine that answers with text depending only on the prompt"""

    def __init__(self,model,**kwargs):
        """Remember the model folder and log the engine settings"""
        self.model = model
        log({"event":"init","model":model,**kwargs})
        # classifiers answer with ids from their exported tokenizer so the real decode path is tested
        path = TOKENIZERS/f"{os.path.basename(model)}.json"
        self.tokenizer = Tokenizer.from_file(str(path)) if path.exists() else None
        # copying a real engine that can exit with an error code after the work is already saved
        if "LBS_STUB_EXIT_CODE" in os.environ:
            atexit.register(os._exit,int(os.environ["LBS_STUB_EXIT_CODE"]))

    def generate(self,prompts,params):
        """Answer every prompt and log the request"""
        log({"event":"generate","model":self.model,"n_prompts":len(prompts),"first_prompt":prompts[0],"params":vars(params)})
        outputs = []
        for p in prompts:
            ids = p["prompt_token_ids"]
            if params.max_tokens==1:
                text = "Yes" if len(ids)%2 else "No"
            elif params.max_tokens==128:
                refusal = "yes" if len(ids)%3==0 else "no"
                text = f"Harmful request: yes\nResponse refusal: {refusal}\nHarmful response: no"
            else:
                # the folder name changes the length so different models give different answers
                count = 3+(len(ids)+len(os.path.basename(self.model)))%40
                text = " ".join(["word"]*count)
            token_ids = self.tokenizer.encode(text,add_special_tokens=False).ids if self.tokenizer else list(range(len(text.split())))
            outputs.append(Output([Completion(text,token_ids,"stop")]))
        return outputs
