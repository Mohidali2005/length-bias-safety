import argparse

from lbs import prompts
from lbs import run_all

parser = argparse.ArgumentParser(description="run every stage of the study in order")
parser.add_argument("--kaggle",action="store_true",help="run the gpu stages that generate and score answers")
parser.add_argument("--resume-from",help="results folder of an earlier kaggle run to continue from")
args = parser.parse_args()

if args.kaggle:
    run_all.run(args.resume_from)
else:
    prompts.build()
