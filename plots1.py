# A slightly modified form of plots.py which is able to generate heatmaps and scatterplots.
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
mdf["PNRA"] = mdf["PInA"]/mdf["NInA"]
mdf["PNRB"] = mdf["PInB"]/mdf["NInB"]

print(mdf.head())
print(mdf.shape)

plt.style.use("seaborn")
plt.figure(figsize=(6,5), dpi=200)

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

#[print(item) for item in iili]

## HEATMAP CODE
##sns.set_style("ticks")
#sns.set_theme()
#sns.set(rc={'figure.figsize':(9,7), 'figure.dpi':115})
##sns.set_context("talk")
#sns.set_context("talk", font_scale=0.6, rc={"lines.linewidth": 1.5})
#pldf = pd.DataFrame(iili, columns=["InA", "InB", "Value"])
#pldf = pldf.pivot('InA', 'InB', 'Value')
##mask = np.zeros((6,6))
##mask[np.tril_indices_from(mask, -1)] = True
#ax = sns.heatmap(pldf, annot=True, linewidth=0.4, cmap="crest", cbar_kws={'label': 'CC AB'})
#ax.invert_yaxis()
#plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95, hspace=0, wspace=0)
##plt.tight_layout()
#plt.show()

## SCATTERPLOT CODE
#mdf = mdf[(mdf["BC A"] >= 0.55) & (mdf["BC B"] >= 0.55)]
#plt.scatter(mdf["InABR"], mdf["CC AB"], c=mdf["BCABR"],marker="o", cmap="Spectral", edgecolor="black", linewidth=0.7)

#cbar = plt.colorbar()
#cbar.set_label("log2(BC A/BC B)", fontname="Roboto")

#plt.xlabel("log2(InA/InB)", fontname="Roboto")
#plt.ylabel("CC AB", fontname="Roboto")
#
#plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95, hspace=0, wspace=0)
###plt.title("E = 2N BC>=0.55")
