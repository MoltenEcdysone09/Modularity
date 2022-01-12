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
    if dfl == "810":
        df["Mean Connectivity"] = ["5N & E:2N"]*100
    elif dfl == "2340":
        df["Mean Connectivity"] = ["20N & E:2N"]*100
    elif dfl in ["1320", "1830"]:
        df["Mean Connectivity"] = ["E:2N"] * 100
    elif dfl in ["820", "1340", "1860", "2380"]:
        df["Mean Connectivity"] = ["E:4N"] * 100
    else:
        df["Mean Connectivity"] = ["E:6N"] * 100
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

sns.set(rc={'figure.figsize':(7,3)})
sns.set_context("paper", rc={"font.weight":'bold',"legend.fontsize":16,"legend.title_fontsize":16,"font.size":16,"axes.titlesize":16,"axes.labelsize":16,"xtick.labelsize":16,"ytick.labelsize":16})
#sns.set_context("paper", rc={"legend.fontsize":12,"font.size":12,"axes.titlesize":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12})
sns.set_style("ticks")

ca = sns.jointplot(data=mdf, x="In Degree of TT", y="f1/f2", hue="Mean Connectivity", hue_order=["5N & E:2N","20N & E:2N", "E:2N" ,"E:4N", "E:6N"], legend=False)
ca.ax_joint.axhline(1, linewidth=3, color='r', linestyle = "--")
plt.legend([],[], frameon=False)

plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0)
plt.savefig("F6Bi.svg",dpi=400)
plt.savefig("F6Bi.png",dpi=400, pad_inches=0)
plt.show()

