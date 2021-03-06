from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from statannotations.Annotator import Annotator
from math import log
import itertools

dfli = ["710", "720", "730", "1220", "1240", "1260", "1730", "1760", "1790", "2240", "2280", "22120"]
#dfli = ["710", "720", "730"]
#dfli = ["710", "1220", "1730", "2240"]
#dfli = ["720", "1240", "1760", "2280"]
#dfli = ["730", "1260", "1790",  "22120"]

mdf = pd.DataFrame()

for dfl in dfli:
    bdf = pd.read_csv("B"+dfl+"TS.csv").iloc[:, 2:]
    cdf = pd.read_csv("C"+dfl+"TS.csv").iloc[:, 2:]
    cpdf = pd.read_csv("CP"+dfl+"TS.csv").iloc[:, 1:]
    ddf = pd.read_csv("D"+dfl+"TS.csv").iloc[:, 1:5]
    #df = pd.concat([cdf, cpdf, bdf], axis = 1)
    df = pd.concat([cdf, cpdf, bdf, ddf], axis = 1)
    if dfl in ["710", "1220", "1730", "2240"]:
        df["Mean Connectivity"] = ["E:2N"] * 100
    elif dfl in ["720", "1240", "1760", "2280"]:
        df["Mean Connectivity"] = ["E:4N"] * 100
    else:
        df["Mean Connectivity"] = ["E:6N"] * 100
    if dfl in ["710", "720", "730"]:
        df["Order"] = ["5N"] * 100
    elif dfl in ["1220", "1240", "1260"]:
        df["Order"] = ["10N"] * 100
    elif dfl in ["1730", "1760", "1790"]:
        df["Order"] = ["15N"] * 100
    else:
        df["Order"] = ["20N"] * 100
    #print(df.head())
    mdf = mdf.append(df)

mdf["BCABR"] = np.log2(mdf["BC A"]/mdf["BC B"])
mdf["INABR"] = np.log2(mdf["InA"]/mdf["InB"])
mdf["FInA"] = mdf["InA"]/(mdf["InA"]+mdf["InB"])
mdf["FInB"] = mdf["InB"]/(mdf["InA"]+mdf["InB"])

print(mdf.head())
print(mdf)
print(mdf.shape)

sns.set(rc={'figure.figsize':(6.0,8)})
sns.set_context("paper", rc={"font.weight":'bold',"legend.fontsize":16,"legend.title_fontsize":16,"font.size":16,"axes.titlesize":16,"axes.labelsize":16,"xtick.labelsize":16,"ytick.labelsize":16})
#sns.set_context("paper", rc={"legend.fontsize":16,"font.size":16,"axes.titlesize":16,"axes.labelsize":16,"xtick.labelsize":16,"ytick.labelsize":16})
sns.set_style("ticks")

meanCon = ["E:2N","E:4N","E:6N"]
norder = ["5N","10N","15N","20N"]
#pairs = list(itertools.product(meanCon, norder))
#pairs = [list(x) for x in itertools.combinations(list(itertools.product),2)]
pairs = [
    [("5N", "E:2N"),("5N","E:4N")], 
    [("5N", "E:4N"),("5N","E:6N")], 
    [("5N", "E:6N"),("5N","E:2N")], 
    [("10N", "E:2N"),("10N","E:4N")], 
    [("10N", "E:4N"),("10N","E:6N")], 
    [("10N", "E:6N"),("10N","E:2N")], 
    [("15N", "E:2N"),("15N","E:4N")], 
    [("15N", "E:4N"),("15N","E:6N")], 
    [("15N", "E:6N"),("15N","E:2N")], 
    [("20N", "E:2N"),("20N","E:4N")], 
    [("20N", "E:4N"),("20N","E:6N")], 
    [("20N", "E:6N"),("20N","E:2N")], 
    ]

#pairs1 = [
#    [("10N", "E:2N"),("10N","E:4N")], 
#    [("10N", "E:4N"),("10N","E:6N")], 
#    [("10N", "E:6N"),("10N","E:2N")], 
#    [("15N", "E:2N"),("15N","E:4N")], 
#    [("15N", "E:4N"),("15N","E:6N")], 
#    [("15N", "E:6N"),("15N","E:2N")], 
#    ]

#pairs = [
#    [("E:2N", "5N"), ("E:2N","10N")],
#    [("E:2N", "10N"), ("E:2N","15N")],
#    [("E:2N", "15N"), ("E:2N","20N")],
#    [("E:4N", "5N"), ("E:4N","10N")],
#    [("E:4N", "10N"), ("E:4N","15N")],
#    [("E:4N", "15N"), ("E:4N","20N")],
#    [("E:6N", "5N"), ("E:6N","10N")],
#    [("E:6N", "10N"), ("E:6N","15N")],
#    [("E:6N", "15N"), ("E:6N","20N")],
#    ]

fig, axs = plt.subplots(2, 1)

cc = sns.boxplot(ax=axs[0],data=mdf, x="Mean Connectivity", y="BC B", hue="Order")
#annotator = Annotator(cc, pairs, data=mdf, x="Order", y="CC AB", hue="Mean Connectivity")
#annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[0].set_ylim(0.2, 1)
axs[0].legend(title="Order",bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


cp = sns.boxplot(ax=axs[1], data=mdf, x="Order", y="BC B", hue="Mean Connectivity")
annotator = Annotator(cp, pairs, data=mdf, x="Order", y="BC B", hue="Mean Connectivity")
annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[1].legend(title="Mean Con.",bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95, hspace=0, wspace=0)
plt.tight_layout()
plt.savefig("S1EF.svg",dpi=400)
plt.savefig("S1EF.png",dpi=400, pad_inches=0)
plt.show()
