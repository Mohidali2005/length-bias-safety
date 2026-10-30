import csv

import matplotlib
import matplotlib.pyplot as plt

from lbs.paths import ANALYSIS

# drawing to files only so no window is needed
matplotlib.use("Agg")

DOSE = ["A","blend25","blend50","blend75","B"]
DOSE_LABELS = ["A","25","50","75","B"]
CONTROLS = [("A_matched","length matched A","darkorange","s"),("A_brief","brief A","seagreen","D")]
# each arrow runs from the long model to the shorter model
ARROWS = [("rto_dpo","rto_rdpo","firebrick"),("rto_dpo","rto_simpo","firebrick"),
          ("pr_mis_dpo","pr_mis_rdpo","darkviolet"),("pr_l3_dpo","pr_l3_simpo","darkviolet")]


def panel(ax,rates,words,value,title):
    """Draw the dose line and the controls and the check pairs of one measure against answer length"""
    xs = [rates[c][words] for c in DOSE]
    ys = [rates[c][value] for c in DOSE]
    errors = [[rates[c][value]-rates[c][value+"_lo"] for c in DOSE],[rates[c][value+"_hi"]-rates[c][value] for c in DOSE]]
    ax.errorbar(xs,ys,yerr=errors,color="steelblue",marker="o",capsize=3,label="A to B blends")
    for x,y,label in zip(xs,ys,DOSE_LABELS):
        ax.annotate(label,(x,y),textcoords="offset points",xytext=(5,5),fontsize=8,color="steelblue")
    for name,label,color,marker in CONTROLS:
        r = rates[name]
        ax.errorbar(r[words],r[value],yerr=[[r[value]-r[value+"_lo"]],[r[value+"_hi"]-r[value]]],color=color,marker=marker,capsize=3,label=label)
    ax.plot(rates["sft"][words],rates["sft"][value],color="gray",marker="x",linestyle="none",label="sft start model")
    for long,short,color in ARROWS:
        # plotting the end points as dots because arrows alone do not widen the axes
        ax.plot([rates[long][words],rates[short][words]],[rates[long][value],rates[short][value]],color=color,marker=".",linestyle="none")
        ax.annotate("",xy=(rates[short][words],rates[short][value]),xytext=(rates[long][words],rates[long][value]),
                    arrowprops={"arrowstyle":"->","color":color})
        ax.annotate(short,(rates[short][words],rates[short][value]),textcoords="offset points",xytext=(4,-10),fontsize=7,color=color)
    ax.plot([],[],color="firebrick",label="RTO pairs")
    ax.plot([],[],color="darkviolet",label="princeton pairs")
    ax.set_xlabel("mean answer length in words on the same prompts")
    ax.set_ylabel(title+" (%)")
    ax.set_title(title)


def draw():
    """Plot harmful compliance and over refusal against answer length for every condition"""
    with open(ANALYSIS/"stats"/"rates.csv",encoding="utf-8",newline="") as f:
        rates = {r["condition"]:{k:float(v) for k,v in r.items() if k!="condition"} for r in csv.DictReader(f)}

    fig,axes = plt.subplots(1,2,figsize=(12,5.5))
    panel(axes[0],rates,"harm_words","comply","Harmful compliance")
    panel(axes[1],rates,"xs_words","refusal","Over refusal")
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=len(labels),fontsize=8,frameon=False)
    fig.tight_layout(rect=(0,0.06,1,1))
    (ANALYSIS/"figures").mkdir(parents=True,exist_ok=True)
    fig.savefig(ANALYSIS/"figures"/"effect.pdf")
    fig.savefig(ANALYSIS/"figures"/"effect.png",dpi=150)
    print(f"figure written with {len(rates)} conditions")

    with open(ANALYSIS/"stats"/"differences.csv",encoding="utf-8",newline="") as f:
        differences = list(csv.DictReader(f))
    fig,axes = plt.subplots(1,2,figsize=(11,4.5))
    ys = [rates[c]["comply"] for c in DOSE]
    errors = [[rates[c]["comply"]-rates[c]["comply_lo"] for c in DOSE],[rates[c]["comply_hi"]-rates[c]["comply"] for c in DOSE]]
    axes[0].bar(["A\n(normal DPO)","25% B","50% B","75% B","B\n(short DPO)"],ys,yerr=errors,capsize=4,color="steelblue")
    for i,y in enumerate(ys):
        axes[0].text(i,4,f"{y:.1f}%",ha="center",color="white",fontsize=9)
    axes[0].set_ylim(0,100)
    axes[0].set_ylabel("harmful prompts answered (%)")
    axes[0].set_title("Harmful compliance drops from A to B")

    # drawing the decisive control in gray because it cannot yet separate training from shorter answers
    colors = ["seagreen","seagreen","seagreen","gray"]
    for i,(row,color) in enumerate(zip(differences,colors)):
        d,lo,hi = float(row["diff"]),float(row["lo"]),float(row["hi"])
        axes[1].errorbar(d,i,xerr=[[d-lo],[hi-d]],color=color,marker="o",capsize=4)
        # placing the number above the point because the axis is inverted
        axes[1].text(d,i-0.15,f"{d:+.1f}",ha="center",color=color,fontsize=9)
    axes[1].axvline(0,color="black",linewidth=0.8)
    axes[1].set_yticks(range(len(differences)),["B vs A\nHarmBench","B vs A\nWildGuard","B vs A\nStrongREJECT","B vs length\nmatched A"])
    # leaving room above the first point so its number is not clipped
    axes[1].set_ylim(len(differences)-0.5,-0.6)
    axes[1].set_xlabel("change in harmfulness, points out of 100 (95% CI)")
    axes[1].set_title("B is a little safer than A on all three scorers")
    fig.text(0.5,0.01,"513 harmful prompts. Over refusal is not shown because A and B refuse none of 250 harmless prompts.\n"
             "Against A answers picked to be closer to B's length the gap shrinks to -1.4 with a CI including 0, "
             "so training and shorter answers cannot yet be told apart.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,0.07,1,1))
    fig.savefig(ANALYSIS/"figures"/"safety_simple.png",dpi=150)
    fig.savefig(ANALYSIS/"figures"/"safety_simple.pdf")


if __name__ == "__main__":
    draw()
