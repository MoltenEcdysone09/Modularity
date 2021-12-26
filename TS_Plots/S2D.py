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


sns.set(rc={'figure.figsize':(5.5,3.5)})
sns.set_context("paper", rc={"font.weight":'bold',"legend.fontsize":9,"legend.title_fontsize":10,"font.size":12,"axes.titlesize":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12})
#sns.set_context("paper", rc={"legend.fontsize":12,"font.size":12,"axes.titlesize":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12})
sns.set_style("ticks")

#iili = []
#
#for s in range(1,7):
#    for d in range(1,7):
#        sdf = mdf[(mdf["InA"]==s) & (mdf["InB"]==d)]
#        if not sdf.empty:
#            ccmean = round(np.mean(list(sdf["BC A"])), 3)
#            #ccstd = round(np.std(list(sdf["CC AB"])), 3)
#            #iili.append([s,d,ccmean,ccstd])
#            iili.append([s,d,ccmean])
#
#ia = [item[0] for item in iili]
#ib = [item[1] for item in iili]
#ccm = [item[2] for item in iili]
#ccmn = [abs(item)*100*5 for item in ccm]
#
#[print(item) for item in iili]

## HEATMAP CODE
#pldf = pd.DataFrame(iili, columns=["InA", "InB", "Value"])
#pldf = pldf.pivot('InA', 'InB', 'Value')
#ax = sns.heatmap(pldf, annot=True, cmap="crest_r", cbar_kws={'label': ''})
#ax.invert_yaxis()
#plt.xlabel("In Degree of B")
#plt.ylabel("In Degree of A")
#plt.title("BC A")
#plt.tight_layout()
#plt.savefig("HMPIABbb.svg",dpi=300)
#plt.savefig("HMPIABbb.png",dpi=300, pad_inches=0)
#plt.show()

# SCATTERPLOT CODE
mdf = mdf[(mdf["BC A"] >= 0.55) & (mdf["BC B"] >= 0.55)]
sns.scatterplot(data=mdf, x="InABR", y="ConPr", hue="Mean Connectivity", hue_order=["5N & E:2N","20N & E:2N", "E:2N" ,"E:4N", "E:6N"], palette=['#4c72b0', '#4c72b0', '#55a868', '#c44e52', '#8172b3'], style="Mean Connectivity", markers=["P","o","o","o","D"])

plt.legend(title="Mean Connectivity", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylim(0.78, 0.98)
plt.xlabel("log2(InA/InB)")
plt.ylabel("Fraction of 01 & 10 States")

##plt.title("E = 2N BC>=0.55")
plt.tight_layout()
plt.savefig("S2D.svg",dpi=400)
plt.savefig("S2D.png",dpi=400, pad_inches=0)
plt.show()

