from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from numpy.polynomial.polynomial import polyfit
from math import log

dfli = ["810", "820", "830", "1320", "1340", "1360", "1830", "1860", "1890", "2340", "2380", "23120"]

mdf = pd.DataFrame()

for dfl in dfli:
    bdf = pd.read_csv("B"+dfl+"TT.csv").iloc[:, 2:]
    cdf = pd.read_csv("C"+dfl+"TT.csv").iloc[:, 2:]
    cpdf = pd.read_csv("CP"+dfl+"TT.csv").iloc[:, 1:]
    ddf = pd.read_csv("D"+dfl+"TT.csv").iloc[:, 1:7]
    idf = pd.read_csv("I"+dfl+"TT.csv").iloc[:, 2:]
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
plt.scatter(mdf["PNRA"], mdf["PNRB"], c=mdf["CC AB"],marker="o", cmap="Spectral", edgecolor="black", linewidth=0.7, alpha=0.8)
cbar = plt.colorbar()
plt.title("CC AB")
plt.xlabel("log2(+ve InA/-ve InA)")
plt.ylabel("log2(+ve InB/-ve InB)")
plt.tight_layout()
plt.savefig("S9Bi.svg",dpi=400)
plt.savefig("S9Bi.png",dpi=400, pad_inches=0)
plt.show()

plt.scatter(mdf["PNRA"], mdf["PNRB"], c=mdf["BCABR"],marker="o", cmap="Spectral", edgecolor="black", linewidth=0.7, alpha=0.8)
cbar = plt.colorbar()
plt.title("log2(BC A/BC B)")
plt.xlabel("log2(+ve InA/-ve InA)")
plt.ylabel("log2(+ve InB/-ve InB)")
plt.tight_layout()
plt.savefig("S9Bii.svg",dpi=400)
plt.savefig("S9Bii.png",dpi=400, pad_inches=0)
plt.show()

plt.scatter(mdf["PNRA"], mdf["PNRB"], c=mdf["ConPr"],marker="o", cmap="Spectral", edgecolor="black", linewidth=0.7, alpha=0.8)
cbar = plt.colorbar()
plt.title("Fraction of Single High States")
plt.xlabel("log2(+ve InA/-ve InA)")
plt.ylabel("log2(+ve InB/-ve InB)")
plt.tight_layout()
plt.savefig("S9Biii.svg",dpi=400)
plt.savefig("S9Biii.png",dpi=400, pad_inches=0)
plt.show()


