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
    #idf = pd.read_csv("I"+dfl+"TS.csv").iloc[:, 2:]
    df = pd.concat([cdf, cpdf, bdf, ddf], axis = 1)
    #print(df.head())
    mdf = mdf.append(df)

mdf["InABR"] = np.log2(mdf["InA"]/mdf["InB"])
mdf["BCABR"] = np.log2(mdf["BC A"]/mdf["BC B"])
#mdf["PNRA"] = mdf["PInA"]/mdf["NInA"]
#mdf["PNRB"] = mdf["PInB"]/mdf["NInB"]

print(mdf.head())
print(mdf.shape)


sns.set(rc={'figure.figsize':(4,3.5)})
sns.set_context("paper", rc={"font.weight":'bold',"legend.fontsize":11,"legend.title_fontsize":11,"font.size":11,"axes.titlesize":11,"axes.labelsize":11,"xtick.labelsize":11,"ytick.labelsize":11})
#sns.set_context("paper", rc={"legend.fontsize":11,"font.size":11,"axes.titlesize":11,"axes.labelsize":11,"xtick.labelsize":11,"ytick.labelsize":11})
sns.set_style("ticks")

iili = []

for s in range(1,7):
    for d in range(1,7):
        sdf = mdf[(mdf["InA"]==s) & (mdf["InB"]==d)]
        if not sdf.empty:
            ccmean = round(np.mean(list(sdf["CC AB"])), 3)
            #ccstd = round(np.std(list(sdf["CC AB"])), 3)
            #iili.append([s,d,ccmean,ccstd])
            iili.append([s,d,ccmean])

ia = [item[0] for item in iili]
ib = [item[1] for item in iili]
ccm = [item[2] for item in iili]
ccmn = [abs(item)*100*5 for item in ccm]
#
#[print(item) for item in iili]

# HEATMAP CODE
pldf = pd.DataFrame(iili, columns=["InA", "InB", "Value"])
pldf = pldf.pivot('InA', 'InB', 'Value')
ax = sns.heatmap(pldf, annot=True, cmap="crest", cbar_kws={'label': ''})
ax.invert_yaxis()
plt.xlabel("In Degree of B")
plt.ylabel("In Degree of A")
plt.title("CC AB")
plt.tight_layout()
plt.savefig("F4Ai.svg",dpi=400)
plt.savefig("F4Ai.png",dpi=400, pad_inches=0)
plt.show()

