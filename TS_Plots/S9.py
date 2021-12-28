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
    idf = pd.read_csv("I"+dfl+"TS.csv").iloc[:, 2:]
    df = pd.concat([cdf, cpdf, bdf, ddf, idf], axis = 1)
    #print(df.head())
    mdf = mdf.append(df)

mdf["InABR"] = np.log2(mdf["InA"]/mdf["InB"])
mdf["BCABR"] = np.log2(mdf["BC A"]/mdf["BC B"])
mdf["PNRA"] = np.log2(mdf["PInA"]/mdf["NInA"])
mdf["PNRB"] = np.log2(mdf["PInB"]/mdf["NInB"])

print(mdf.head())
print(mdf.shape)

sns.set(rc={'figure.figsize':(4.5,3.5)})
sns.set_context("paper", rc={"font.weight":'bold',"legend.fontsize":12,"legend.title_fontsize":12,"font.size":12,"axes.titlesize":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12})
#sns.set_context("paper", rc={"legend.fontsize":12,"font.size":12,"axes.titlesize":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12})
sns.set_style("ticks")

# SCATTERPLOT CODE
#mdf = mdf[(mdf["BC A"] >= 0.55) & (mdf["BC B"] >= 0.55)]
plt.scatter(mdf["PNRA"], mdf["PNRB"], c=mdf["CC AB"],marker="o", cmap="Spectral", edgecolor="black", linewidth=0.7, alpha=0.8)
cbar = plt.colorbar()

plt.title("CC AB")
plt.xlabel("log2(+ve InA/-ve InA)")
plt.ylabel("log2(+ve InB/-ve InB)")

##plt.title("E = 2N BC>=0.55")
plt.tight_layout()
plt.savefig("S9A.svg",dpi=400)
plt.savefig("S9A.png",dpi=400, pad_inches=0)
plt.show()

plt.scatter(mdf["PNRA"], mdf["PNRB"], c=mdf["BCABR"],marker="o", cmap="Spectral", edgecolor="black", linewidth=0.7, alpha=0.8)
cbar = plt.colorbar()

plt.title("log2(BC A/BC B)")
plt.xlabel("log2(+ve InA/-ve InA)")
plt.ylabel("log2(+ve InB/-ve InB)")

##plt.title("E = 2N BC>=0.55")
plt.tight_layout()
plt.savefig("S9B.svg",dpi=400)
plt.savefig("S9B.png",dpi=400, pad_inches=0)
plt.show()

plt.scatter(mdf["PNRA"], mdf["PNRB"], c=mdf["ConPr"],marker="o", cmap="Spectral", edgecolor="black", linewidth=0.7, alpha=0.8)
cbar = plt.colorbar()

plt.title("Fraction of 01 & 10 States")
plt.xlabel("log2(+ve InA/-ve InA)")
plt.ylabel("log2(+ve InB/-ve InB)")

##plt.title("E = 2N BC>=0.55")
plt.tight_layout()
plt.savefig("S9C.svg",dpi=400)
plt.savefig("S9C.png",dpi=400, pad_inches=0)
plt.show()












