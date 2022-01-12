from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from numpy.polynomial.polynomial import polyfit
from math import log2

dfli = ["810", "820", "830", "1320", "1340", "1360", "1830", "1860", "1890", "2340", "2380", "23120"]
#dfli = ["710", "720", "730"]
#dfli = ["710", "1220", "1730", "2240"]
#dfli = ["720", "1240", "1760", "2280"]
#dfli = ["730", "1260", "1790",  "22120"]

mdf = pd.DataFrame()

for dfl in dfli:
    cpdf = pd.read_csv("CP"+dfl+"TT.csv").iloc[:, 1:]
    cpdf.rename(columns={"ConPr":"f1"}, inplace=True)
    cp1df = pd.read_csv("CP"+dfl+"TT1.csv").iloc[:, 1:]
    cp1df.rename(columns={"ConPr":"f2"}, inplace=True)
    cp0df = pd.read_csv("CP"+dfl+"TT0.csv").iloc[:, 1:]
    cp0df.rename(columns={"ConPr":"f3"}, inplace=True)
    ddf = pd.read_csv("D"+dfl+"TT.csv").iloc[:, 1:7]
    df = pd.concat([ddf, cpdf, cp1df, cp0df], axis = 1)
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

mdf["f1/f2"] = mdf["f1"]/mdf["f2"]
mdf["f1/f3"] = mdf["f1"]/mdf["f3"]
mdf["InA/InB"] = mdf["InA"]/mdf["InB"]
mdf["InB/InC"] = mdf["InB"]/mdf["InC"]
mdf["InC/InA"] = mdf["InC"]/mdf["InA"]
mdf["In Degree of TT"] = mdf["InA"] + mdf["InB"] + mdf["InC"]

print(mdf.shape)
print(mdf)
print(mdf.shape)

sns.set(rc={'figure.figsize':(10.5,8)})
sns.set_context("paper", rc={"font.weight":'bold',"legend.fontsize":12,"legend.title_fontsize":12,"font.size":12,"axes.titlesize":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12})
#sns.set_context("paper", rc={"legend.fontsize":12,"font.size":12,"axes.titlesize":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12})
sns.set_style("ticks")

#sns.lineplot(data=mdf, x="Mean Connectivity", y="f1/f2", style="Order", hue="Order", err_style= "bars", markers=True, dashes=False,err_kws={'capsize':3} )
#sns.pointplot(data=mdf, x="Mean Connectivity", y="f1/f2", style="Order", hue="Order", ci="sd", capsize=0.1)
#sns.boxplot(data=mdf, x="Mean Connectivity", y="f1/f2", hue="Order")
#sns.violinplot(data=mdf, x="Order", y="f1/f2", hue="Mean Connectivity")
#sns.scatterplot(data=mdf, x="In TT", y="f1/f2")
#sns.scatterplot(data=mdf, x="PNRA", y="ConPr", hue="Mean Connectivity")
#sns.displot(data=mdf, x="CC AB", hue="Order", kind="kde", fill="true")
#sns.jointplot(data=mdf, x="INABR", y="CC AB", hue="H/M/L INABR", xlim=[-2.5,2.5], palette=palette, hue_order=order, edgecolor="black", alpha=0.8)
#sns.scatterplot(data=mdf, x="InC/InA", y="f1/f2", hue="Mean Connectivity")
#sns.jointplot(data=mdf, x="InA/InB", y="f1/f2", hue="Mean Connectivity")
sns.jointplot(data=mdf, x="In Degree of TT", y="f1/f2", hue="Mean Connectivity")
#plt.ylabel("log2(InA/InB)")
plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0)
plt.savefig("JPf1f2in.svg",dpi=300)
plt.savefig("JPf1f2in.png",dpi=300, pad_inches=0)
plt.show()

