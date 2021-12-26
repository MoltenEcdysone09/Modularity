from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from statannotations.Annotator import Annotator
from math import log
import itertools

mdf = pd.DataFrame()

crng = [1,2,3,4,5,6,7,8,9,12,13]

for dfl in crng:
    bdf = pd.read_csv("Bimodality C"+str(dfl)+".csv").iloc[:, 2:]
    cdf = pd.read_csv("CorrelationNP C"+str(dfl)+".csv").iloc[:, 2:]
    fdf = pd.read_csv("C"+str(dfl)+"f1f2.csv").iloc[:,1:]
    ldf = pd.read_csv("MTLC"+str(dfl)+".csv").iloc[:, 2:]
    #df = pd.concat([cdf, cpdf, bdf], axis = 1)
    df = pd.concat([cdf, bdf, fdf, ldf], axis = 1)
    if dfl == 1:
        df["Type"] = ["TT"] * 100
    else:
        df["Type"] = ["C"+str(dfl)] * 100
    mdf = mdf.append(df)
mdf = mdf.rename({"C1":"TT"}, axis="columns")

mdf["f1/f2"] = mdf["f1"]/mdf["f2"]
mdf["Multi Stable"] = 1 - mdf["Mono"]

print(mdf.head())
print(mdf)
print(mdf.shape)

ttli = ["TT"]
rndli = ["C"+str(x) for x in [2,3,4,5,6,7,8,9,12,13]]

pairs = list(itertools.product(ttli, rndli))


#sns.set(rc={'figure.figsize':(13,10)})
sns.set_context("paper", rc={"font.size":10, "font.weight":'bold'})
sns.set(font_scale=1.1)
sns.set_style("ticks")

fig, axs = plt.subplots(2, 3, figsize=(17, 10))
# 0,0
cc = sns.boxplot(ax=axs[0,0],data=mdf, x="Type", y="CC(BC)")
#annotator = Annotator(cc, pairs, data=mdf, x="Type", y="CC(BC)")
#annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[0,0].set_ylim(-0.92, 1.1)
axs[0,0].set_ylabel("CC BC")
axs[0,0].legend([],[], frameon=False)
# 0,1
cc = sns.boxplot(ax=axs[0,1],data=mdf, x="Type", y="CC(CA)")
#annotator = Annotator(cc, pairs, data=mdf, x="Type", y="CC(CA)")
#annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[0,1].set_ylim(-0.92, 1.1)
axs[0,1].set_ylabel("CC CA")
axs[0,1].legend([],[], frameon=False)
# 0,2
cc = sns.boxplot(ax=axs[0,2],data=mdf, x="Type", y="BC A")
#annotator = Annotator(cc, pairs, data=mdf, x="Type", y="BC A")
#annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[0,2].set_ylim(0.2, 1)
axs[0,2].legend([],[], frameon=False)
# 1,0
cc = sns.boxplot(ax=axs[1,0],data=mdf, x="Type", y="BC B")
#annotator = Annotator(cc, pairs, data=mdf, x="Type", y="BC B")
#annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[1,0].set_ylim(0.2, 1)
axs[1,0].legend([],[], frameon=False)
# 1,1
cc = sns.boxplot(ax=axs[1,1],data=mdf, x="Type", y="BC C")
#annotator = Annotator(cc, pairs, data=mdf, x="Type", y="BC C")
#annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[1,1].set_ylim(0.2, 1)
axs[1,1].legend([],[], frameon=False)
# 1,2
cg = sns.boxplot(ax=axs[1,2],data=mdf, x="Type", y="Multi Stable")
#annotator = Annotator(cg, pairs, data=mdf, x="Type", y="Multi Stable")
#annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[1,2].set_ylim(0.1, 1)
axs[1,2].legend([],[], frameon=False)

#sns.lineplot(data=mdf, x="NetP", y="CC AB", style="NetS", hue="NetS", err_style= "bars", markers=True, dashes=False,err_kws={'capsize':3} )
#sns.pointplot(data=mdf, x="NetP", y="BC B", style="NetS", hue="NetS", ci="sd", capsize=0.1)
#sns.boxplot(data=mdf, x="Type", y="Multi Stable")
#sns.violinplot(data=mdf, x="Type", y="CC(AB)")
#sns.violinplot(data=mdf, x="NetS", y="ConPr", hue="NetP")
#sns.scatterplot(data=mdf, x="CC AB", y="ConPr", hue="INABR")
#sns.scatterplot(data=mdf, x="PNRA", y="ConPr", hue="NetP")
#sns.displot(data=mdf, x="BC A", hue="Type", kind="kde", fill="true")
#sns.jointplot(data=mdf, x="INABR", y="CC AB", hue="H/M/L INABR", xlim=[-2.5,2.5], palette=palette, hue_order=order, edgecolor="black", alpha=0.8)
#plt.ylabel("Fraction of 001/010/100 States")
#plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95, hspace=0, wspace=0)
#plt.ylabel("Fraction of Single High Steady States")
#plt.ylabel("CC CA")
#plt.legend(loc="upper right", title="Order")
plt.tight_layout()
plt.savefig("SFig8.svg",dpi=300)
plt.savefig("SFig8.png",dpi=300, pad_inches=0)
plt.show()
#print(np.mean(mdf[(mdf["NetP"] == "E:4N")]["InA"]))
#print(np.std(mdf[(mdf["NetP"] == "E:4N")]["InA"]))
#print(np.mean(mdf[(mdf["NetP"] == "E:4N")]["InB"]))
#print(np.std(mdf[(mdf["NetP"] == "E:4N")]["InB"]))
