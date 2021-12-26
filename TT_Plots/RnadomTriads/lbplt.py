from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from numpy.polynomial.polynomial import polyfit
from math import log


mdf = pd.DataFrame()

crng = [1,2,3,4,5,6,7,8,9,12,13]

for dfl in crng:
    bdf = pd.read_csv("Bimodality C"+str(dfl)+".csv").iloc[:, 2:]
    cdf = pd.read_csv("CorrelationNP C"+str(dfl)+".csv").iloc[:, 2:]
    fdf = pd.read_csv("C"+str(dfl)+"f1f2.csv").iloc[:,1:]
    ldf = pd.read_csv("MTLC"+str(dfl)+".csv").iloc[:, 2:]
    #df = pd.concat([cdf, cpdf, bdf], axis = 1)
    df = pd.concat([cdf, bdf, fdf, ldf], axis = 1)
    df["Type"] = ["C"+str(dfl)] * 100
    mdf = mdf.append(df)

mdf["f1/f2"] = mdf["f1"]/mdf["f2"]
mdf["Multi Stable"] = 1 - mdf["Mono"]

print(mdf.head())
print(mdf)
print(mdf.shape)

sns.set(rc={'figure.figsize':(5,4)})
sns.set_context("paper", rc={"font.size":24, "font.weight":'bold'})
#sns.set_context("paper")
sns.set_style("ticks")


#palette = sns.color_palette(["#f03b20", "#feb24c", "#ffeda0", "#2ca25f", "#99d8c9", "#e5f5f9", "#c51b8a", "#fa9fb5", "#fde0dd"]) 
#palette = sns.color_palette(["#de2d26", "#fc9272", "#fee0d2", "#3182bd", "#9ecae1", "#deebf7", "#31a354", "#a1d99b", "#e5f5e0"]) 

#sns.lineplot(data=mdf, x="NetP", y="CC AB", style="NetS", hue="NetS", err_style= "bars", markers=True, dashes=False,err_kws={'capsize':3} )
#sns.pointplot(data=mdf, x="NetP", y="BC B", style="NetS", hue="NetS", ci="sd", capsize=0.1)
sns.boxplot(data=mdf, x="Type", y="Multi Stable")
#sns.violinplot(data=mdf, x="Type", y="CC(AB)")
#sns.violinplot(data=mdf, x="NetS", y="ConPr", hue="NetP")
#sns.scatterplot(data=mdf, x="CC AB", y="ConPr", hue="INABR")
#sns.scatterplot(data=mdf, x="PNRA", y="ConPr", hue="NetP")
#sns.displot(data=mdf, x="BC A", hue="Type", kind="kde", fill="true")
#sns.jointplot(data=mdf, x="INABR", y="CC AB", hue="H/M/L INABR", xlim=[-2.5,2.5], palette=palette, hue_order=order, edgecolor="black", alpha=0.8)
#plt.ylabel("Fraction of 001/010/100 States")
#plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95, hspace=0, wspace=0)
#plt.ylabel("Fraction of Single High Steady States")
#plt.ylabel("CC CA")
#plt.legend(loc="upper right", title="Order")
plt.tight_layout()
plt.savefig("BPMlt.svg",dpi=300)
plt.savefig("BPMlt.png",dpi=300, pad_inches=0)
plt.show()
#print(np.mean(mdf[(mdf["NetP"] == "E:4N")]["InA"]))
#print(np.std(mdf[(mdf["NetP"] == "E:4N")]["InA"]))
#print(np.mean(mdf[(mdf["NetP"] == "E:4N")]["InB"]))
#print(np.std(mdf[(mdf["NetP"] == "E:4N")]["InB"]))
