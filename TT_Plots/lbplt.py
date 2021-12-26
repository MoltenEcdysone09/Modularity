from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from numpy.polynomial.polynomial import polyfit
from math import log

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

sns.set(rc={'figure.figsize':(5,4)})
sns.set_context("paper", rc={"font.size":24, "font.weight":'bold'})
#sns.set_context("paper")
sns.set_style("ticks")


#palette = sns.color_palette(["#f03b20", "#feb24c", "#ffeda0", "#2ca25f", "#99d8c9", "#e5f5f9", "#c51b8a", "#fa9fb5", "#fde0dd"]) 
#palette = sns.color_palette(["#de2d26", "#fc9272", "#fee0d2", "#3182bd", "#9ecae1", "#deebf7", "#31a354", "#a1d99b", "#e5f5e0"]) 

sns.lineplot(data=mdf, x="Mean Connectivity", y="CC AB", style="Order", hue="Order", err_style= "bars", markers=True, dashes=False,err_kws={'capsize':3} )
#sns.pointplot(data=mdf, x="Mean Connectivity", y="BC B", style="Order", hue="Order", ci="sd", capsize=0.1)
#sns.boxplot(data=mdf, x="Mean Connectivity", y="ConPr", hue="Order")
#sns.violinplot(data=mdf, x="Order", y="ConPr", hue="Mean Connectivity")
#sns.scatterplot(data=mdf, x="CC AB", y="ConPr", hue="INABR")
#sns.scatterplot(data=mdf, x="PNRA", y="ConPr", hue="Mean Connectivity")
#sns.displot(data=mdf, x="CC AB", hue="Order", kind="kde", fill="true")
#sns.jointplot(data=mdf, x="INABR", y="CC AB", hue="H/M/L INABR", xlim=[-2.5,2.5], palette=palette, hue_order=order, edgecolor="black", alpha=0.8)
#plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95, hspace=0, wspace=0)
plt.ylabel("Fraction of Single High Steady States")
#plt.ylabel("CC CA")
#plt.legend(loc="upper right", title="Order")
plt.tight_layout()
plt.savefig("LPf1.svg",dpi=300)
plt.savefig("LPf1.png",dpi=300, pad_inches=0)
plt.show()
