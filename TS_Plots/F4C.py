from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from numpy.polynomial.polynomial import polyfit
from math import log

dfli = ["710", "720", "730", "1220", "1240", "1260", "1730", "1760", "1790", "2240", "2280", "22120"]
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
    if dfl == "710":
        df["Mean Connectivity"] = ["5N & E:2N"]*100
    elif dfl == "2240":
        df["Mean Connectivity"] = ["20N & E:2N"]*100
    elif dfl in ["1220", "1730"]:
        df["Mean Connectivity"] = ["E:2N"] * 100
    elif dfl in ["720", "1240", "1760", "2280"]:
        df["Mean Connectivity"] = ["E:4N"] * 100
    else:
        df["Mean Connectivity"] = ["E:6N"] * 100
    #print(df.head())
    mdf = mdf.append(df)

mdf["InABR"] = np.log2(mdf["InA"]/mdf["InB"])
mdf["BCABR"] = np.log2(mdf["BC A"]/mdf["BC B"])
#mdf["PNRA"] = mdf["PInA"]/mdf["NInA"]
#mdf["PNRB"] = mdf["PInB"]/mdf["NInB"]


print(mdf.head())
print(mdf.shape)


sns.set(rc={'figure.figsize':(4.5,3.5)})
sns.set_context("paper", rc={"font.weight":'bold',"legend.fontsize":9,"legend.title_fontsize":10,"font.size":12,"axes.titlesize":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12})
#sns.set_context("paper", rc={"legend.fontsize":12,"font.size":12,"axes.titlesize":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12})
sns.set_style("ticks")


# SCATTERPLOT CODE
mdf = mdf[(mdf["BC A"] >= 0.55) & (mdf["BC B"] >= 0.55)]
sns.scatterplot(data=mdf, x="InABR", y="CC AB", hue="Mean Connectivity", hue_order=["5N & E:2N","20N & E:2N", "E:2N" ,"E:4N", "E:6N"], palette=['#4c72b0', '#4c72b0', '#55a868', '#c44e52', '#8172b3'], style="Mean Connectivity", markers=["P","o","o","o","D"])

plt.legend(loc="upper left", title="Mean Connectivity")
plt.xlabel("log2(InA/InB)")
plt.ylabel("CC AB")
plt.tight_layout()
plt.savefig("F4C.svg",dpi=400)
plt.savefig("F4C.png",dpi=400, pad_inches=0)
plt.show()

