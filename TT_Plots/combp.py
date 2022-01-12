from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from numpy.polynomial.polynomial import polyfit
from math import log

dfli = ["810", "820", "830", "1320", "1340", "1360", "1830", "1860", "1890", "2340", "2380", "23120"]
#dfli = ["810", "1220", "1830", "2240"]
#dfli = ["820", "1240", "1860", "2280"]
#dfli = ["830", "1260", "1890",  "22120"]

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
mdf["InBCR"] = np.log2(mdf["InB"]/mdf["InC"])
mdf["InCAR"] = np.log2(mdf["InC"]/mdf["InA"])
mdf["BCABR"] = np.log2(mdf["BC A"]/mdf["BC B"])
mdf["BCBCR"] = np.log2(mdf["BC B"]/mdf["BC C"])
mdf["BCCAR"] = np.log2(mdf["BC C"]/mdf["BC A"])

print(mdf.head())
print(mdf.shape)

sns.set(rc={'figure.figsize':(5,4)})
#sns.set_context("paper", rc={"font.size":24, "font.weight":'bold'})
sns.set_context("paper")
sns.set_style("ticks")

iili = []

for s in range(1,8):
    for d in range(1,8):
        sdf = mdf[(mdf["InC"]==s) & (mdf["InA"]==d)]
        if not sdf.empty:
            ccmean = round(np.mean(list(sdf["ConPr"])), 3)
            #ccstd = round(np.std(list(sdf["CC AC"])), 3)
            #iili.append([s,d,ccmean,ccstd])
            iili.append([s,d,ccmean])

ia = [item[0] for item in iili]
ib = [item[1] for item in iili]
ccm = [item[2] for item in iili]
ccmn = [abs(item)*100*5 for item in ccm]

#[print(item) for item in iili]

# HEATMAP CODE
pldf = pd.DataFrame(iili, columns=["InC", "InA", "Value"])
pldf = pldf.pivot('InC', 'InA', 'Value')
#mask = np.zeros((6,6))
#mask[np.tril_indices_from(mask, -1)] = True
ax = sns.heatmap(pldf, annot=True, cmap="crest", cbar_kws={'label': 'Fraction of Single High Steady States'})
ax.invert_yaxis()
plt.xlabel("In Degree of A")
plt.ylabel("In Degree of C")
plt.tight_layout()
plt.savefig("HMPICAcp.svg",dpi=300)
plt.savefig("HMPICAcp.png",dpi=300, pad_inches=0)
plt.show()

## SCATTERPLOT CODE
##mdf = mdf[(mdf["BC A"] >= 0.55) & (mdf["BC B"] >= 0.55)]
#plt.scatter(mdf["InABR"], mdf["CC AB"], c=mdf["BCABR"],marker="o", cmap="Spectral", edgecolor="black", linewidth=0.8, alpha=0.8)
#cbar = plt.colorbar()
#cbar.set_label("log2(BC A/BC B)")
#
#plt.xlabel("log2(InA/InB)")
#plt.ylabel("CC AB")
#
###plt.title("E = 2N BC>=0.55")
#plt.tight_layout()
#plt.savefig("SPLABcc55.svg",dpi=300)
#plt.savefig("SPLABcc55.png",dpi=300, pad_inches=0)
#plt.show()