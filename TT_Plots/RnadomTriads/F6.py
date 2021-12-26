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
    ddf = pd.read_csv("DC"+str(dfl)+".csv").iloc[:,1:]
    #df = pd.concat([cdf, cpdf, bdf], axis = 1)
    df = pd.concat([cdf, bdf, fdf, ldf,ddf], axis = 1)
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

sns.set(rc={'figure.figsize':(5,3.5)})
sns.set_context("paper", rc={"font.weight":'bold',"legend.fontsize":13,"legend.title_fontsize":13,"font.size":13,"axes.titlesize":13,"axes.labelsize":13,"xtick.labelsize":13,"ytick.labelsize":13})
#sns.set_context("paper", rc={"legend.fontsize":13,"font.size":13,"axes.titlesize":13,"axes.labelsize":13,"xtick.labelsize":13,"ytick.labelsize":13})
sns.set_style("ticks")

sns.boxplot(data=mdf, x="Type", y="CC(AB)")
plt.ylabel("CC AB")
plt.ylim(-0.9, 1.1)
plt.tight_layout()
plt.savefig("F6A.svg",dpi=400)
plt.savefig("F6A.png",dpi=400, pad_inches=0)
plt.show()

sns.boxplot(data=mdf, x="Type", y="f1/f2")
plt.ylabel("f1/f2")
plt.ylim(0.2, 4.2)
plt.tight_layout()
plt.savefig("F6B.svg",dpi=400)
plt.savefig("F6B.png",dpi=400, pad_inches=0)
plt.show()
