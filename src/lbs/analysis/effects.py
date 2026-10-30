import csv

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar

from lbs import config
from lbs import records
from lbs.analysis.first_checks import PAIRS
from lbs.analysis.first_checks import RESAMPLES
from lbs.analysis.first_checks import resample
from lbs.paths import ANALYSIS
from lbs.paths import RUN

HARMFUL = ["harmbench","strongreject"]
DOSE = {"A":0.0,"blend25":0.25,"blend50":0.5,"blend75":0.75,"B":1.0}
# scorer and its field followed by the short name used in this file
FIELDS = [("harmbench","comply","comply"),("wildguard","refusal","refusal"),("wildguard","harmful_response","harmful"),
          ("strmatch","refusal","str_refusal"),("strongreject","score","score")]
TABLE = ["A","blend25","blend50","blend75","B","A_matched","A_sample4_mean","A_brief","sft",
         "rto_dpo","rto_rdpo","rto_simpo","pr_mis_dpo","pr_mis_rdpo","pr_l3_dpo","pr_l3_simpo"]


def load(name):
    """Collect the answer length and every score of one condition keyed by set and prompt id and sample"""
    rows = {(r["set"],r["id"],r["sample"]):{"words":len(r["response"].split())} for r in records.read_jsonl(RUN/"gen"/f"{name}.jsonl")}
    for scorer,field,short in FIELDS:
        path = RUN/"scores"/scorer/f"{name}.jsonl"
        if path.exists():
            for r in records.read_jsonl(path):
                rows[(r["set"],r["id"],r["sample"])][short] = r[field]
    assert all(v is not None and v!=-1 for row in rows.values() for v in row.values()),f"{name} has an invalid or unparsed score"
    return rows


def column(rows,keys,field):
    """Pull one field for the given prompts out of a condition as a float array"""
    return np.array([rows[k][field] for k in keys],dtype=float)


def paired(x,y):
    """Return the mean paired difference of two arrays with its bootstrap interval over prompts"""
    d = x-y
    lo,hi = np.percentile(d[resample(len(d))].mean(1),[2.5,97.5])
    return d.mean(),lo,hi


def rate(x):
    """Return the mean of one array with its bootstrap interval over prompts"""
    lo,hi = np.percentile(x[resample(len(x))].mean(1),[2.5,97.5])
    return x.mean(),lo,hi


def show(e,digits=1):
    """Format an estimate and its interval with signs"""
    return f"{e[0]:+.{digits}f} [{e[1]:+.{digits}f}, {e[2]:+.{digits}f}]"


def level(e):
    """Format a rate and its interval without signs"""
    return f"{e[0]:.1f} [{e[1]:.1f}, {e[2]:.1f}]"


def excludes(e):
    """Say whether an interval leaves out zero"""
    return e[1]>0 or e[2]<0


def exact(x,y):
    """Count the discordant prompts of two paired yes or no arrays and give the exact mcnemar p value"""
    b,c = int(((x==1)&(y==0)).sum()),int(((x==0)&(y==1)).sum())
    p = mcnemar([[0,b],[c,0]],exact=True).pvalue if b+c else 1.0
    return f"{b} / {c}, p = {p:.2g}"


def stacked(data,levels,keys,field):
    """Stack several conditions on the same prompts into one long table for a model clustered by prompt"""
    y = np.concatenate([column(data[c],keys,field) for c in levels])
    x = np.concatenate([np.full(len(keys),v) for v in levels.values()])
    words = np.concatenate([column(data[c],keys,"words") for c in levels])
    groups = np.tile(np.arange(len(keys)),len(levels))
    return y,x,words,groups


def logit(y,x,words,groups):
    """Fit a logistic model of y on x without and with log words and return the x coefficient with its clustered interval for both"""
    # following the usual rule of ten events per variable because rarer outcomes give separated and meaningless fits
    if min(y.sum(),(1-y).sum())<10:
        return None,None
    fits = []
    for X in [x[:,None],np.column_stack([x,np.log(words)])]:
        fit = sm.GLM(y,sm.add_constant(X),family=sm.families.Binomial()).fit(cov_type="cluster",cov_kwds={"groups":groups})
        lo,hi = fit.conf_int()[1]
        fits.append((fit.params[1],lo,hi))
    return fits


def slope(data,keys,field):
    """Fit a straight line of the rate in percent against the share of B over the dose conditions and bootstrap its slope"""
    share = np.array(list(DOSE.values()))
    s = share-share.mean()
    Y = np.array([100*column(data[c],keys,field) for c in DOSE])
    idx = resample(len(keys))
    boots = (s@np.array([y[idx].mean(1) for y in Y]))/(s@s)
    lo,hi = np.percentile(boots,[2.5,97.5])
    return (s@Y.mean(1))/(s@s),lo,hi


def write():
    """Run every paired comparison and control of the main analysis and write the tables"""
    data = {name:load(name) for name in config.CONDITIONS}
    harm = [k for k in data["A"] if k[0] in HARMFUL]
    xs = [k for k in data["A"] if k[0]=="xstest_safe"]
    pre = [k for k in data["A"] if k[0]=="prefill"]

    # picking for every prompt the A sample closest in words to the answer of B and the lowest sample on ties
    sample4 = data["A_sample4"]
    data["A_matched"] = {}
    data["A_sample4_mean"] = {}
    for s,i,_ in harm+xs:
        b = data["B"][(s,i,0)]["words"]
        best = min(range(4),key=lambda j:abs(sample4[(s,i,j)]["words"]-b))
        data["A_matched"][(s,i,0)] = sample4[(s,i,best)]
        data["A_sample4_mean"][(s,i,0)] = {f:np.mean([sample4[(s,i,j)][f] for j in range(4)]) for f in sample4[(s,i,0)]}

    lines = ["# Main analysis","",
             f"Run `{RUN.parent.name}`. Rates are percent. Differences are percentage points with paired bootstrap 95% intervals "
             f"over prompts ({RESAMPLES:,} resamples, seed {config.SEED}). McNemar gives discordant prompts as "
             "(first yes and second no) / (first no and second yes) with the exact p value. Logistic coefficients are log odds with "
             "95% intervals clustered by prompt.","",
             "Harmful prompts are harmbench + strongreject (513). Over refusal is the WildGuard refusal rate on xstest_safe (250). "
             "A_matched is for every prompt the A_sample4 answer (temperature 0.7) closest in words to B's answer. "
             "A_sample4_mean averages the four samples and shows what sampling alone does.",""]

    lines += ["## Rates per condition","",
              "| Condition | Words harmful | HarmBench compliance | WildGuard harmful | WildGuard refusal harmful | String refusal harmful "
              "| Median words of refusals | Prefill compliance | StrongREJECT score (0 to 100) | Words xstest | Over refusal | String refusal xstest |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    csv_rows = []
    for c in TABLE:
        rows = data[c]
        refused = [rows[k]["words"] for k in harm if rows[k]["refusal"]==1]
        has_prefill = pre[0] in rows
        has_score = "score" in rows[harm[0]]
        comply = rate(100*column(rows,harm,"comply"))
        over = rate(100*column(rows,xs,"refusal"))
        cells = [f"{column(rows,harm,'words').mean():.0f}",level(comply),level(rate(100*column(rows,harm,"harmful"))),
                 level(rate(100*column(rows,harm,"refusal"))),level(rate(100*column(rows,harm,"str_refusal"))),
                 f"{np.median(refused):.0f}" if refused else "none",
                 level(rate(100*column(rows,pre,"comply"))) if has_prefill else "not run",
                 level(rate(100*column(rows,harm,"score"))) if has_score else "not run",
                 f"{column(rows,xs,'words').mean():.0f}",level(over),level(rate(100*column(rows,xs,"str_refusal")))]
        lines.append(f"| {c} | "+" | ".join(cells)+" |")
        csv_rows.append({"condition":c,"harm_words":column(rows,harm,"words").mean(),"comply":comply[0],"comply_lo":comply[1],"comply_hi":comply[2],
                         "xs_words":column(rows,xs,"words").mean(),"refusal":over[0],"refusal_lo":over[1],"refusal_hi":over[2]})
    lines.append("")

    length = [k for k in data["A"] if k[0]=="length"]
    gap = paired(column(data["B"],length,"words"),column(data["A"],length,"words"))
    lines += ["## Point 1: B is shorter than A","",
              f"Words on the length prompts, B minus A: {show(gap)}. Percent shorter and token counts are in first_checks.md.",""]

    lines += ["## Point 2: B against A and against the length controls",""]
    main_sign = {}
    for label,keys,field in [("Harmful compliance (HarmBench)",harm,"comply"),("Over refusal (WildGuard on xstest_safe)",xs,"refusal")]:
        a,b,m = [column(data[c],keys,field) for c in ["A","B","A_matched"]]
        brief = column(data["A_brief"],keys,field)
        sampled = column(data["A_sample4_mean"],keys,field)
        ba,bm,ma = paired(100*b,100*a),paired(100*b,100*m),paired(100*m,100*a)
        fits = logit(*stacked(data,{"A":0,"B":1},keys,field))
        main_sign[field] = np.sign(ba[0])
        same = lambda e:np.sign(e[0])==np.sign(ba[0])
        a_holds = excludes(bm) and same(bm)
        b_holds = fits[1] is not None and excludes(fits[1]) and same(fits[1])
        if a.sum()==0 and b.sum()==0:
            verdict = "cannot be tested: neither A nor B has this outcome on any prompt"
        elif not excludes(ba):
            verdict = "does not hold: the B minus A interval includes zero"
        elif a_holds and b_holds:
            verdict = "holds: both length controls leave a difference in the same direction"
        elif a_holds or b_holds:
            verdict = f"partial: only control {'(a)' if a_holds else '(b)'} leaves a difference in the same direction"
        else:
            verdict = "does not hold: neither length control leaves a difference in the same direction"
        opposite = [name for name,e in [("(a)",bm),("(b)",fits[1])] if e is not None and excludes(e) and not same(e)]
        if opposite and excludes(ba):
            verdict += f". Control {' and '.join(opposite)} points the opposite way with an interval excluding zero"
        words = [column(data[c],keys,"words").mean() for c in ["A","B","A_matched","A_brief"]]
        lines += [f"### {label}","",
                  "| Comparison | Difference | McNemar |","|---|---|---|",
                  f"| B minus A | {show(ba)} | {exact(b,a)} |",
                  f"| (a) B minus A_matched (decisive) | {show(bm)} | {exact(b,m)} |",
                  f"| A_matched minus A | {show(ma)} | {exact(m,a)} |",
                  f"| A_sample4_mean minus A (sampling alone) | {show(paired(100*sampled,100*a))} | |",
                  f"| A_brief minus A | {show(paired(100*brief,100*a))} | {exact(brief,a)} |",
                  f"| B minus A_brief | {show(paired(100*b,100*brief))} | {exact(b,brief)} |",""]
        lines.append(f"Mean words on these prompts: A {words[0]:.0f}, B {words[1]:.0f}, A_matched {words[2]:.0f}, A_brief {words[3]:.0f}. "
                     f"Matching closes {100*(words[0]-words[2])/(words[0]-words[1]):.0f}% of the word gap between A and B "
                     f"and the brief prompt closes {100*(words[0]-words[3])/(words[0]-words[1]):.0f}%.")
        lines.append("")
        if fits[0] is None:
            lines.append("(b) Logistic model of B against A: not estimable because fewer than 10 answers of A and B have the rarer outcome.")
        else:
            lines.append(f"(b) Logistic coefficient of B against A: {show(fits[0],2)} without length and {show(fits[1],2)} with log words "
                         f"(odds ratio {np.exp(fits[1][0]):.2f}).")
        lines += ["",f"Point 2 for {label.lower()}: **{verdict}**. Control (a) can over control because refusals are short, "
                  "so the A sample closest in length to a refusal of B is often a refusal of A.",""]

    lines += ["## Point 3: the change follows the share of B","",
              "Slope of a straight line through A, blend25, blend50, blend75 and B in percentage points per full step from A to B, "
              "bootstrapped over prompts. The logistic slope is clustered by prompt.","",
              "| Measure | Rates A, 25, 50, 75, B | Bootstrap slope | Slope excludes zero | Logistic slope | Logistic slope with log words |",
              "|---|---|---|---|---|---|"]
    for label,keys,field in [("Harmful compliance",harm,"comply"),("Over refusal",xs,"refusal"),
                             ("WildGuard harmful",harm,"harmful"),("Prefill compliance",pre,"comply")]:
        rates = ", ".join(f"{100*column(data[c],keys,field).mean():.1f}" for c in DOSE)
        fits = logit(*stacked(data,DOSE,keys,field))
        cells = ["not estimable","not estimable"] if fits[0] is None else [show(fits[0],2),show(fits[1],2)]
        trend = slope(data,keys,field)
        lines.append(f"| {label} | {rates} | {show(trend)} | {'yes' if excludes(trend) else 'no'} | {cells[0]} | {cells[1]} |")
    lines.append("")

    lines += ["## Second and third checks","",
              "Each shorter model against its long reference on the same prompts. Direction is compared with B minus A.","",
              "| Pair | Compliance difference | McNemar | Logistic with log words | WildGuard harmful difference | Over refusal difference | McNemar "
              "| Compliance direction against B vs A |",
              "|---|---|---|---|---|---|---|---|"]
    for short,long in PAIRS:
        cs,cl = column(data[short],harm,"comply"),column(data[long],harm,"comply")
        rs,rl = column(data[short],xs,"refusal"),column(data[long],xs,"refusal")
        ws,wl = column(data[short],harm,"harmful"),column(data[long],harm,"harmful")
        diff = paired(100*cs,100*cl)
        fits = logit(*stacked(data,{long:0,short:1},harm,"comply"))
        direction = ("same" if np.sign(diff[0])==main_sign["comply"] else "opposite")+(" and excludes zero" if excludes(diff) else " but includes zero")
        lines.append(f"| {short} vs {long} | {show(diff)} | {exact(cs,cl)} | {show(fits[1],2)} | {show(paired(100*ws,100*wl))} | "
                     f"{show(paired(100*rs,100*rl))} | {exact(rs,rl)} | {direction} |")
    lines.append("")

    lines += ["## Extras for B against A","","| Measure | B minus A | McNemar |","|---|---|---|"]
    for label,keys,field in [("WildGuard harmful response",harm,"harmful"),("WildGuard refusal on harmful prompts",harm,"refusal"),
                             ("String refusal on harmful prompts",harm,"str_refusal"),("Prefill compliance",pre,"comply"),
                             ("String refusal on xstest_safe",xs,"str_refusal")]:
        a,b = column(data["A"],keys,field),column(data["B"],keys,field)
        lines.append(f"| {label} | {show(paired(100*b,100*a))} | {exact(b,a)} |")
    a,b,brief = [column(data[c],harm,"score") for c in ["A","B","A_brief"]]
    lines.append(f"| StrongREJECT score (0 to 100) | {show(paired(100*b,100*a))} | |")
    lines.append(f"| StrongREJECT score A_brief minus A | {show(paired(100*brief,100*a))} | |")
    lines.append("")

    lines += ["## Agreement between HarmBench compliance and WildGuard harmful response","",
              "Not a validation against people. If the two scorers agree much less on one model than the other that hints at a length effect "
              "in a scorer. The hand labels decide.","","| Condition | Agreement | Cohen kappa |","|---|---|---|"]
    # leaving out the averaged samples because their values are not yes or no
    for c in [c for c in TABLE if c!="A_sample4_mean"]:
        x,y = column(data[c],harm,"comply"),column(data[c],harm,"harmful")
        agree = (x==y).mean()
        chance = x.mean()*y.mean()+(1-x.mean())*(1-y.mean())
        lines.append(f"| {c} | {100*agree:.1f} | {(agree-chance)/(1-chance):.2f} |")
    lines.append("")

    # collecting the headline differences in percentage points for the simple figure
    differences = []
    for label,first,second,field in [("HarmBench","B","A","comply"),("WildGuard","B","A","harmful"),("StrongREJECT","B","A","score"),
                                     ("HarmBench vs length matched A","B","A_matched","comply")]:
        e = paired(100*column(data[first],harm,field),100*column(data[second],harm,field))
        differences.append({"label":label,"diff":e[0],"lo":e[1],"hi":e[2]})

    (ANALYSIS/"stats").mkdir(parents=True,exist_ok=True)
    (ANALYSIS/"stats"/"effects.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
    with open(ANALYSIS/"stats"/"differences.csv","w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(differences[0]))
        writer.writeheader()
        writer.writerows({k:(f"{v:.4f}" if isinstance(v,float) else v) for k,v in row.items()} for row in differences)
    with open(ANALYSIS/"stats"/"rates.csv","w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows({k:(f"{v:.4f}" if isinstance(v,float) else v) for k,v in row.items()} for row in csv_rows)
    print(f"main analysis written for {len(TABLE)} conditions")


if __name__ == "__main__":
    write()
