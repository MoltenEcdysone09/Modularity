from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from numpy.polynomial.polynomial import polyfit
from math import log

df = pd.read_csv("NORMtt.txt")
#df = df.iloc[:200, :]

#For converting to steady states
hlli = []
for x,y,z in zip(df["A"], df["B"],df["C"]):
    if x > 0:
        xv = "1"
    else:
        xv = "0"
    if y > 0:
        yv = "1"
    else:
        yv = "0"
    if z > 0:
        zv = "1"
    else:
        zv = "0"
    hlli.append(xv+yv+zv)

#hlli = []
#for x,y in zip(df["A"], df["B"]):
#    if x > 0:
#        xv = "1"
#    else:
#        xv = "0"
#    if y > 0:
#        yv = "1"
#    else:
#        yv = "0"
#    hlli.append(xv+yv)

df["Steady States"] = hlli

print(df)

sns.set(rc={'figure.figsize':(10.5,8)})
sns.set_context("paper", rc={"font.weight":'bold',"legend.fontsize":16,"legend.title_fontsize":16,"font.size":16,"axes.titlesize":16,"axes.labelsize":16,"xtick.labelsize":16,"ytick.labelsize":16})
#sns.set_context("paper", rc={"legend.fontsize":16,"font.size":16,"axes.titlesize":16,"axes.labelsize":16,"xtick.labelsize":16,"ytick.labelsize":16})
sns.set_style("ticks")

#dist = list(df["A"]) + list(df["B"] + list(df["C"]))

#sns.displot(x=dist, color="#bd5e57", kind="kde", fill=True)
sns.displot(data=df, x="A", y="B", cmap="crest", cbar=True, kind="kde", fill=True)
plt.ylim(-3, 2.5)
plt.xlim(-3, 2.5)
#sns.regplot(data=df, x="B", y="C", color="#bd5e57")
#sns.countplot(data=df, x="Steady States", order=["10","01","00","11"])
#sns.countplot(data=df, x="Steady States", order=["001","010", "100", "011","101","110","111","000"])
#plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95, hspace=0, wspace=0)
#sns.despine()
plt.tight_layout()
plt.savefig("F1E.svg",dpi=400)
plt.savefig("F1E.png",dpi=400, pad_inches=0)
plt.show()
