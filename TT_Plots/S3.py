from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from statannotations.Annotator import Annotator
from math import log
import itertools

dfli = ["810", "820", "830", "1320", "1340", "1360", "1830", "1860", "1890", "2340", "2380", "23120"]
#dfli = ["710", "720", "730"]
#dfli = ["710", "1220", "1730", "2240"]
#dfli = ["720", "1240", "1760", "2280"]
#dfli = ["730", "1260", "1790",  "22120"]

mdf = pd.DataFrame()

for dfl in dfli:
    bdf = pd.read_csv("B"+dfl+"TT.csv").iloc[:, 2:]
    cdf = pd.read_csv("C"+dfl+"TT.csv").iloc[:, 2:]
    cpdf = pd.read_csv("CP"+dfl+"TT.csv").iloc[:, 1:]
    ddf = pd.read_csv("D"+dfl+"TT.csv").iloc[:, 1:7]
    idf = pd.read_csv("I"+dfl+"TT.csv").iloc[:, 1:8]
    #df = pd.concat([cdf, cpdf, bdf], axis = 1)
    df = pd.concat([cdf, cpdf, bdf, ddf, idf], axis = 1)
    if dfl in ["810", "1320", "1830", "2340"]:
        df["Mean Connectivity"] = ["E:2N"] * 100
    elif dfl in ["820", "1340", "1860", "2380"]:
        df["Mean Connectivity"] = ["E:4N"] * 100
    else:
        df["Mean Connectivity"] = ["E:6N"] * 100
    if dfl in ["810", "820", "830"]:
        df["Order"] = ["5N"] * 100
    elif dfl in ["1320", "1340", "1360"]:
        df["Order"] = ["10N"] * 100
    elif dfl in ["1830", "1860", "1890"]:
        df["Order"] = ["15N"] * 100
    else:
        df["Order"] = ["20N"] * 100
    #print(df.head())
    mdf = mdf.append(df)

mdf["BCABR"] = np.log2(mdf["BC A"]/mdf["BC B"])
mdf["INABR"] = np.log2(mdf["InA"]/mdf["InB"])
mdf["BCACR"] = np.log2(mdf["BC A"]/mdf["BC C"])
mdf["INACR"] = np.log2(mdf["InA"]/mdf["InC"])
mdf["BCBCR"] = np.log2(mdf["BC B"]/mdf["BC C"])
mdf["INBCR"] = np.log2(mdf["InB"]/mdf["InC"])

print(mdf.head())
print(mdf)
print(mdf.shape)

sns.set(rc={'figure.figsize':(10.5,8)})
sns.set_context("paper", rc={"font.weight":'bold',"legend.fontsize":13,"legend.title_fontsize":13,"font.size":13,"axes.titlesize":13,"axes.labelsize":13,"xtick.labelsize":13,"ytick.labelsize":13})
#sns.set_context("paper", rc={"legend.fontsize":13,"font.size":13,"axes.titlesize":13,"axes.labelsize":13,"xtick.labelsize":13,"ytick.labelsize":13})
sns.set_style("ticks")


#meanCon = ["E:2N","E:4N","E:6N"]
#norder = ["5N","10N","15N","20N"]
#pairs = list(itertools.product(meanCon, norder))
#pairs = [list(x) for x in itertools.combinations(list(itertools.product),2)]
pairs1 = [
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

pairs2 = [
    [("E:2N", "5N"), ("E:2N","10N")],
    [("E:2N", "10N"), ("E:2N","15N")],
    [("E:2N", "15N"), ("E:2N","20N")],
    [("E:4N", "5N"), ("E:4N","10N")],
    [("E:4N", "10N"), ("E:4N","15N")],
    [("E:4N", "15N"), ("E:4N","20N")],
    [("E:6N", "5N"), ("E:6N","10N")],
    [("E:6N", "10N"), ("E:6N","15N")],
    [("E:6N", "15N"), ("E:6N","20N")],
    ]

fig, axs = plt.subplots(2, 4, figsize=(14, 7))


ba = sns.boxplot(ax=axs[0,0], data=mdf, x="Mean Connectivity", y="CC BC", hue="Order")
annotator = Annotator(ba, pairs2, data=mdf, x="Mean Connectivity", y="CC BC", hue="Order")
annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[0,0].legend([],[], frameon=False)


bc = sns.boxplot(ax=axs[0,1], data=mdf, x="Mean Connectivity", y="CC AC", hue="Order")
annotator = Annotator(bc, pairs2, data=mdf, x="Mean Connectivity", y="CC AC", hue="Order")
annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[0,1].legend([],[], frameon=False)


bb = sns.boxplot(ax=axs[0,2], data=mdf, x="Mean Connectivity", y="BC B", hue="Order")
annotator = Annotator(bb, pairs2, data=mdf, x="Mean Connectivity", y="BC B", hue="Order")
annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[0,2].legend([],[], frameon=False)


br = sns.boxplot(ax=axs[0,3], data=mdf, x="Mean Connectivity", y="BC C", hue="Order")
annotator = Annotator(br, pairs2, data=mdf, x="Mean Connectivity", y="BC C", hue="Order")
annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[0,3].legend(title="Order",bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
#axs[0,1].legend([],[], frameon=False)


cc = sns.boxplot(ax=axs[1,0],data=mdf, x="Order", y="CC BC", hue="Mean Connectivity")
annotator = Annotator(cc, pairs1, data=mdf, x="Order", y="CC BC", hue="Mean Connectivity")
annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[1,0].legend([],[], frameon=False)


cp = sns.boxplot(ax=axs[1,1], data=mdf, x="Order", y="CC AC", hue="Mean Connectivity")
annotator = Annotator(cp, pairs1, data=mdf, x="Order", y="CC AC", hue="Mean Connectivity")
annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[1,1].legend([],[], frameon=False)


cb = sns.boxplot(ax=axs[1,2],data=mdf, x="Order", y="BC B", hue="Mean Connectivity")
annotator = Annotator(cb, pairs1, data=mdf, x="Order", y="BC B", hue="Mean Connectivity")
annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[1,2].legend([],[], frameon=False)

cb = sns.boxplot(ax=axs[1,3],data=mdf, x="Order", y="BC C", hue="Mean Connectivity")
annotator = Annotator(cb, pairs1, data=mdf, x="Order", y="BC C", hue="Mean Connectivity")
annotator.configure(test='Mann-Whitney', text_format='star').apply_and_annotate()
axs[1,3].legend(title="Mean Con.",bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
#axs[1,2].legend([],[], frameon=False)

plt.tight_layout()
plt.savefig("S3.svg",dpi=400)
plt.savefig("S3.png",dpi=400, pad_inches=0)
plt.show()
