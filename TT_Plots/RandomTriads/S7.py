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

sns.set(rc={'figure.figsize':(4,3.5)})
sns.set_context("paper", rc={"font.weight":'bold',"legend.fontsize":13,"legend.title_fontsize":13,"font.size":13,"axes.titlesize":13,"axes.labelsize":13,"xtick.labelsize":13,"ytick.labelsize":13})
#sns.set_context("paper", rc={"legend.fontsize":13,"font.size":13,"axes.titlesize":13,"axes.labelsize":13,"xtick.labelsize":13,"ytick.labelsize":13})
sns.set_style("ticks")

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
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

plt.tight_layout()
plt.savefig("S7.svg",dpi=400)
plt.savefig("S7.png",dpi=400, pad_inches=0)
plt.show()
