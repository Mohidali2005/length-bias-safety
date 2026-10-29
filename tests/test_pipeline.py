import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

from lbs import config
from lbs import records
from lbs.scoring import tokens


def run(env,*extra):
    """Run the orchestrator in a fresh process and fail with its output if it crashes"""
    result = subprocess.run([sys.executable,"-m","lbs.run_all",*extra],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def read_log(folder):
    """Read the stub engine log of one run"""
    path = folder/"stub_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def generations(log):
    """Keep the generate calls that produced answers and not scores"""
    return [e for e in log if e["event"]=="generate" and e["params"]["max_tokens"]==config.MAX_NEW_TOKENS]


def manifest(folder):
    """Read the manifest of one run"""
    return json.loads((folder/"results"/"manifest.json").read_text(encoding="utf-8"))


def test_every_condition_and_scorer_finishes(full_run):
    """A full run generates and scores everything and leaves no model on disk"""
    m = manifest(full_run)
    assert set(m["conditions"].values())=={"done"}
    assert set(m["scorers"].values())=={"done"}
    assert m["generated"]==list(config.CONDITIONS)
    for scorer,settings in config.SCORERS.items():
        assert m["scored"][scorer]==settings["conditions"]
    assert list((full_run/"models").iterdir())==[]


def test_answer_and_score_rows_line_up(full_run):
    """Every condition has one answer per prompt and sample and every scorer scores exactly its sets"""
    gen = full_run/"results"/"gen"
    for name,c in config.CONDITIONS.items():
        rows = records.read_jsonl(gen/f"{name}.jsonl")
        assert len(rows)==5*len(c["sets"])*c["samples"]
        assert sorted({r["sample"] for r in rows})==list(range(c["samples"]))
        assert {r["set"] for r in rows}==set(c["sets"])
        assert all(r["prefill"]==("Sure, here is" if r["set"]=="prefill" else None) for r in rows)
    for scorer,settings in config.SCORERS.items():
        for name in settings["conditions"]:
            rows = [r for r in records.read_jsonl(gen/f"{name}.jsonl") if r["set"] in settings["sets"]]
            scores = records.read_jsonl(full_run/"results"/"scores"/scorer/f"{name}.jsonl")
            assert [(r["set"],r["id"],r["sample"]) for r in rows]==[(s["set"],s["id"],s["sample"]) for s in scores]


def test_classifier_answers_decode_to_valid_labels(full_run):
    """Classifier answers go through the decode and parse path without turning into invalid labels"""
    scores = full_run/"results"/"scores"
    for name in config.SCORERS["harmbench"]["conditions"]:
        assert all(s["comply"] in [0,1] for s in records.read_jsonl(scores/"harmbench"/f"{name}.jsonl"))
    for name in config.SCORERS["wildguard"]["conditions"]:
        assert not any(s["parse_error"] for s in records.read_jsonl(scores/"wildguard"/f"{name}.jsonl"))


def test_brief_control_uses_the_mean_words_of_B(full_run):
    """The brief instruction asks for the rounded mean word count of B on the same sets"""
    gen = full_run/"results"/"gen"
    b = [r for r in records.read_jsonl(gen/"B.jsonl") if r["set"] in config.SAFETY]
    words = round(statistics.mean(len(r["response"].split()) for r in b))
    assert json.loads((gen/"A_brief.done.json").read_text())["brief_words"]==words
    for r in records.read_jsonl(gen/"A_brief.jsonl"):
        assert r["message"]==r["prompt"]+config.BRIEF.format(words=words)
    for r in records.read_jsonl(gen/"A.jsonl"):
        assert r["message"]==r["prompt"]


def test_engine_settings(full_run):
    """Every engine and sampling setting reaches vllm exactly as configured"""
    log = read_log(full_run)
    events = generations(log)
    assert len(events)==len(config.CONDITIONS)
    for name,event in zip(config.CONDITIONS,events):
        c = config.CONDITIONS[name]
        model = config.MODELS[c["model"]]
        assert Path(event["model"]).name==c["model"]
        assert event["n_prompts"]==5*len(c["sets"])*c["samples"]
        assert event["params"]=={"n":1,"temperature":c["temperature"],"top_p":1.0,"repetition_penalty":1.0,"max_tokens":config.MAX_NEW_TOKENS,"stop_token_ids":model["stop"]}
        ids = event["first_prompt"]["prompt_token_ids"]
        bos = 1 if c["model"].startswith("pr_mis") else 128000
        assert ids.count(bos)==int(model["bos"])
        assert (ids[0]==bos)==model["bos"]
    for e in [e for e in log if e["event"]=="init"]:
        assert (e["dtype"],e["tensor_parallel_size"],e["gpu_memory_utilization"],e["seed"])==("float16",2,0.9,0)
        name = Path(e["model"]).name
        assert e["max_model_len"]==(4096 if name in ["harmbench_cls","wildguard"] else 2048)
        assert "tokenizer_mode" not in e
    classifier = [e for e in log if e["event"]=="generate" and Path(e["model"]).name=="harmbench_cls"]
    assert classifier and all(e["params"]["temperature"]==0.0 and e["params"]["max_tokens"]==1 for e in classifier)
    ids = classifier[0]["first_prompt"]["prompt_token_ids"]
    assert ids[0]==1 and tokens.load("harmbench_cls").decode(ids,skip_special_tokens=True).startswith("[INST] You are a classifier for red teaming")
    judge = [e for e in log if e["event"]=="generate" and Path(e["model"]).name=="wildguard"]
    assert judge and all(e["params"]["temperature"]==0.0 and e["params"]["top_p"]==1.0 and e["params"]["max_tokens"]==128 for e in judge)
    ids = judge[0]["first_prompt"]["prompt_token_ids"]
    # keeping special tokens because inst is a control token in this vocabulary
    assert ids[0]==1 and ids[7]==3 and tokens.load("wildguard").decode(ids,skip_special_tokens=False).startswith("<s> <|user|>\n[INST] You are a classifier for analyzing")


def test_resume_redoes_only_lost_conditions(full_run,tmp_path,make_env):
    """After losing two conditions a rerun redoes those two and touches nothing else"""
    results = tmp_path/"results"
    shutil.copytree(full_run/"results",results)
    lost = ["blend75","pr_l3_simpo"]
    for name in lost:
        for f in results.rglob(f"{name}.*"):
            f.unlink()
    before = {f:f.stat().st_mtime_ns for f in results.rglob("*.jsonl")}
    run(make_env(tmp_path))
    assert [Path(e["model"]).name for e in generations(read_log(tmp_path))]==lost
    assert all(f.stat().st_mtime_ns==t for f,t in before.items())
    for f in (full_run/"results").rglob("*.jsonl"):
        assert f.read_text(encoding="utf-8")==(results/f.relative_to(full_run/"results")).read_text(encoding="utf-8")
    assert set(manifest(tmp_path)["conditions"].values())=={"done"}


def test_resume_from_an_earlier_run(full_run,tmp_path,make_env):
    """Starting from copied results generates and scores nothing again"""
    run(make_env(tmp_path),"--resume-from",str(full_run/"results"))
    assert [e for e in read_log(tmp_path) if e["event"]=="generate"]==[]
    m = manifest(tmp_path)
    assert set(m["conditions"].values())=={"done"}
    assert set(m["scorers"].values())=={"nothing to score"}


def test_deadline_stops_new_work(tmp_path,make_env):
    """With no time left nothing is started and the manifest says why"""
    run(make_env(tmp_path),"--gen-hours","0","--hard-hours","0")
    assert read_log(tmp_path)==[]
    m = manifest(tmp_path)
    assert set(m["conditions"].values())=={"skipped because the generation deadline passed"}
    assert m["generated"]==[]


def test_failures_are_reported_and_noisy_exits_still_count(tmp_path,make_env,local_models):
    """Broken models are reported while engines that exit with an error after saving still count as done"""
    models = tmp_path/"local_models"
    shutil.copytree(local_models,models)
    shutil.rmtree(models/"sft")
    for f in (models/"pr_mis_rdpo").glob("tokenizer*"):
        f.unlink()
    env = make_env(tmp_path,models)
    env["LBS_STUB_EXIT_CODE"] = "3"
    run(env)
    m = manifest(tmp_path)
    assert m["conditions"]["sft"]=="failed while fetching the model"
    assert m["conditions"]["pr_mis_rdpo"]=="failed while generating"
    assert all(status=="done" for name,status in m["conditions"].items() if name not in ["sft","pr_mis_rdpo"])
    assert set(m["scorers"].values())=={"done"}
    assert "sft" not in m["scored"]["harmbench"] and "pr_mis_rdpo" not in m["scored"]["harmbench"]
