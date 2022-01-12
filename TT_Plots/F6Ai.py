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

sns.set(rc={'figure.figsize':(4,3.5)})
sns.set_context("paper", rc={"font.weight":'bold',"legend.fontsize":11,"legend.title_fontsize":11,"font.size":11,"axes.titlesize":11,"axes.labelsize":11,"xtick.labelsize":11,"ytick.labelsize":11})
#sns.set_context("paper", rc={"legend.fontsize":12,"font.size":12,"axes.titlesize":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12})
sns.set_style("ticks")

iili = []

for s in range(1,8):
    for d in range(1,8):
        sdf = mdf[(mdf["InA"]==s) & (mdf["InB"]==d)]
        if not sdf.empty:
            ccmean = round(np.mean(list(sdf["CC AB"])), 3)
            #ccstd = round(np.std(list(sdf["CC AC"])), 3)
            #iili.append([s,d,ccmean,ccstd])
            iili.append([s,d,ccmean])

ia = [item[0] for item in iili]
ib = [item[1] for item in iili]
ccm = [item[2] for item in iili]
ccmn = [abs(item)*100*5 for item in ccm]

#[print(item) for item in iili]

# HEATMAP CODE
pldf = pd.DataFrame(iili, columns=["InA", "InB", "Value"])
pldf = pldf.pivot('InA', 'InB', 'Value')
#mask = np.zeros((6,6))
#mask[np.tril_indices_from(mask, -1)] = True
ax = sns.heatmap(pldf, annot=True, cmap="crest", vmin=-0.6, vmax=-0.1, cbar_kws={'label': ''})
ax.invert_yaxis()
plt.xlabel("In Degree of B")
plt.ylabel("In Degree of A")
plt.title("CC AB")
plt.tight_layout()
plt.savefig("F6Ai.svg",dpi=400)
plt.savefig("F6Ai.png",dpi=400, pad_inches=0)
plt.show()
