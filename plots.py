# Gneral script to generate polts. The first for loop compiles and labels all the file BC, CC, Steady State Fraction and Degree files into a single dataframe.
# This particular script is for TT but can be made to work with TS with minor changes in code.
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from numpy.polynomial.polynomial import polyfit
from math import log

dfli = ["810", "820", "830", "1320", "1340", "1360", "1830", "1860", "1890", "2340", "2380", "23120"]
#dfli = ["710", "720", "730"]
#dfli = ["710", "1220", "1730", "2240"]
#dfli = ["720", "1240", "1760", "2280"]
#dfli = ["730", "1260", "1790",  "22120"]

mdf = pd.DataFrame()

for dfl in dfli:
    bdf = pd.read_csv("B"+dfl+"TT.csv").iloc[:, 2:]
    cdf = pd.read_csv("C"+dfl+"TT.csv").iloc[:, 2:]
    cpdf = pd.read_csv("CP"+dfl+"TT.csv").iloc[:, 1:]
    ddf = pd.read_csv("D"+dfl+"TT.csv").iloc[:, 1:7]
    idf = pd.read_csv("I"+dfl+"TT.csv").iloc[:, 1:8]
    #df = pd.concat([cdf, cpdf, bdf], axis = 1)
    df = pd.concat([cdf, cpdf, bdf, ddf, idf], axis = 1)
    if dfl in ["810", "1320", "1830", "2340"]:
        df["NetP"] = ["E:2N"] * 100
    elif dfl in ["820", "1340", "1860", "2380"]:
        df["NetP"] = ["E:4N"] * 100
    else:
        df["NetP"] = ["E:6N"] * 100
    if dfl in ["810", "820", "830"]:
        df["NetS"] = ["5N"] * 100
    elif dfl in ["1320", "1340", "1360"]:
        df["NetS"] = ["10N"] * 100
    elif dfl in ["1830", "1860", "1890"]:
        df["NetS"] = ["15N"] * 100
    else:
        df["NetS"] = ["20N"] * 100
    #print(df.head())
    mdf = mdf.append(df)

mdf["BCABR"] = np.log2(mdf["BC A"]/mdf["BC B"])
mdf["INABR"] = np.log2(mdf["InA"]/mdf["InB"])
mdf["BCACR"] = np.log2(mdf["BC A"]/mdf["BC C"])
mdf["INACR"] = np.log2(mdf["InA"]/mdf["InC"])
mdf["BCBCR"] = np.log2(mdf["BC B"]/mdf["BC C"])
mdf["INBCR"] = np.log2(mdf["InB"]/mdf["InC"])
mdf["InTT"] = mdf["InA"]+mdf["InB"]+mdf["InC"]
mdf["FInA"] = mdf["InA"]/(mdf["InA"]+mdf["InB"]+mdf["InC"])
mdf["FInB"] = mdf["InB"]/(mdf["InA"]+mdf["InB"]+mdf["InC"])
mdf["FInC"] = mdf["InC"]/(mdf["InA"]+mdf["InB"]+mdf["InC"])
mdf["PNRA"] = mdf["PInA"]/mdf["NInA"]
mdf["PNRB"] = mdf["PInB"]/mdf["NInB"]
mdf["PNRC"] = mdf["PInC"]/mdf["NInC"]

print(mdf.head())
print(mdf)
print(mdf.shape)

#sns.set_style("ticks")
sns.set_theme()
sns.set(rc={'figure.figsize':(9,7), 'figure.dpi':115})
#sns.set_context("talk")
sns.set_context("talk", font_scale=0.9, rc={"lines.linewidth": 2})


#palette = sns.color_palette(["#f03b20", "#feb24c", "#ffeda0", "#2ca25f", "#99d8c9", "#e5f5f9", "#c51b8a", "#fa9fb5", "#fde0dd"]) 
#palette = sns.color_palette(["#de2d26", "#fc9272", "#fee0d2", "#3182bd", "#9ecae1", "#deebf7", "#31a354", "#a1d99b", "#e5f5e0"]) 
#order = ["InABR:L NetP:E:2N", "InABR:M NetP:E:2N", "InABR:H NetP:E:2N","InABR:L NetP:E:4N", "InABR:M NetP:E:4N", "InABR:H NetP:E:4N", "InABR:L NetP:E:6N", "InABR:M NetP:E:6N", "InABR:H NetP:E:6N"]


#sns.lineplot(data=mdf, x="NetP", y="CC AB", style="NetS", hue="NetS", err_style= "bars", markers=True, dashes=False,err_kws={'capsize':3} )
#sns.pointplot(data=mdf, x="NetP", y="BC B", style="NetS", hue="NetS", ci="sd", capsize=0.1)
sns.boxplot(data=mdf, x="NetP", y="ConPr", hue="NetS")
#sns.violinplot(data=mdf, x="NetS", y="ConPr", hue="NetP")
#sns.scatterplot(data=mdf, x="CC AB", y="ConPr", hue="INABR")
#sns.scatterplot(data=mdf, x="PNRA", y="ConPr", hue="NetP")
#sns.displot(data=mdf, x="CC AB", hue="NetS", kind="kde", fill="true")
#sns.jointplot(data=mdf, x="INABR", y="CC AB", hue="H/M/L INABR", xlim=[-2.5,2.5], palette=palette, hue_order=order, edgecolor="black", alpha=0.8)
plt.ylabel("Fraction of 001/010/100 States")
plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95, hspace=0, wspace=0)
plt.show()
