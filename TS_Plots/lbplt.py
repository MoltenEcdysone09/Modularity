from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
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

sns.set(rc={'figure.figsize':(5,4)})
sns.set_context("paper", rc={"font.size":24, "font.weight":'bold'})
#sns.set_context("paper")
sns.set_style("ticks")

#palette = sns.color_palette(["#f03b20", "#feb24c", "#ffeda0", "#2ca25f", "#99d8c9", "#e5f5f9", "#c51b8a", "#fa9fb5", "#fde0dd"]) 
#palette = sns.color_palette(["#de2d26", "#fc9272", "#fee0d2", "#3182bd", "#9ecae1", "#deebf7", "#31a354", "#a1d99b", "#e5f5e0"]) 
#palette = sns.color_palette(['#78c679','#31a354','#006837', '#fc8d59','#e34a33','#b30000', '#41b6c4','#2c7fb8','#253494'])
#order = ["InABR:L Mean Connectivity:E:2N", "InABR:M Mean Connectivity:E:2N", "InABR:H Mean Connectivity:E:2N","InABR:L Mean Connectivity:E:4N", "InABR:M Mean Connectivity:E:4N", "InABR:H Mean Connectivity:E:4N", "InABR:L Mean Connectivity:E:6N", "InABR:M Mean Connectivity:E:6N", "InABR:H Mean Connectivity:E:6N"]

#sns.lineplot(data=mdf, x="Mean Connectivity", y="CC AB", style="Order", hue="Order", err_style= "bars", markers=True, dashes=False,err_kws={'capsize':3} )
#sns.pointplot(data=mdf, x="Mean Connectivity", y="BC B", style="Order", hue="Order", ci="sd", capsize=0.1)
ax = sns.boxplot(data=mdf, x="Mean Connectivity", y="ConPr", hue="Order")
#sns.violinplot(data=mdf, x="Order", y="ConPr", hue="Mean Connectivity")
#sns.scatterplot(data=mdf, x="CC AB", y="ConPr", hue="INABR", palette="vlag")
#sns.scatterplot(data=mdf, x="INABR", y="CC AB", hue="Mean Connectivity")
#sns.displot(data=mdf, x="CC AB", hue="H/M/L INABR", kind="kde", fill="true")
#sns.jointplot(data=mdf, x="INABR", y="CC AB", hue="H/M/L INABR", xlim=[-4,4], palette=palette, hue_order=order, edgecolor="black", alpha=0.8)
plt.ylim(0, 1.2)
plt.ylabel("Fraction of 01 and 10 Steady States")
#plt.legend(loc="upper right", title="Mean Connectivity")
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
#plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95, hspace=0, wspace=0)
plt.tight_layout()
plt.savefig("BPCPo1.svg",dpi=300)
plt.savefig("BPCPo1.png",dpi=300, pad_inches=0)
plt.show()
