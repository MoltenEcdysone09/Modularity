from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from numpy.polynomial.polynomial import polyfit
from math import log

df = pd.read_csv("NORMtt.txt")
df = df.iloc[:300, :]

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

sns.set(rc={'figure.figsize':(4.5,3.5)})
sns.set_context("paper", rc={"font.weight":'bold',"legend.fontsize":14,"legend.title_fontsize":14,"font.size":14,"axes.titlesize":14,"axes.labelsize":14,"xtick.labelsize":14,"ytick.labelsize":14})
#sns.set_context("paper", rc={"legend.fontsize":14,"font.size":14,"axes.titlesize":14,"axes.labelsize":14,"xtick.labelsize":14,"ytick.labelsize":14})
sns.set_style("ticks")

#dist = list(df["A"]) + list(df["B"] + list(df["C"]))

sns.regplot(data=df, x="A", y="B", color="#bd5e57", truncate=False)
#sns.countplot(data=df, x="Steady States", order=["10","01","00","11"])
#sns.countplot(data=df, x="Steady States", order=["001","010", "100", "011","101","110","111","000"])
#plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95, hspace=0, wspace=0)
#sns.despine()
plt.tight_layout()
plt.savefig("S1A.svg",dpi=400)
plt.savefig("S1A.png",dpi=400, pad_inches=0)
plt.show()
