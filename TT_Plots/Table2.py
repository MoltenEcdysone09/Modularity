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
    #idf = pd.read_csv("I"+dfl+"TT.csv").iloc[:, 2:]
    df = pd.concat([cdf, cpdf, bdf, ddf], axis = 1)
    #print(df.head())
    mdf = mdf.append(df)

mdf["InABR"] = np.log2(mdf["InA"]/mdf["InB"])
mdf["InBCR"] = np.log2(mdf["InB"]/mdf["InC"])
mdf["InCAR"] = np.log2(mdf["InC"]/mdf["InA"])
mdf["BCABR"] = np.log2(mdf["BC A"]/mdf["BC B"])
mdf["BCBCR"] = np.log2(mdf["BC B"]/mdf["BC C"])
mdf["BCCAR"] = np.log2(mdf["BC C"]/mdf["BC A"])
mdf["InTT"] = mdf["InA"] + mdf["InB"] + mdf["InC"]

print(mdf.head())
print(mdf.shape)

degli = ["InA","InB","InC","InTT"]
metli = ["ConPr","CC AB","CC BC", "CC AC","BC A","BC B", "BC C"]
ctsli = []

mdf = mdf.dropna(subset=["InA"])

for d in degli:
    tcli = [d]
    for f in metli:
        print([d,f])
        rho, pval = stats.spearmanr(list(mdf[d]),list(mdf[f]))
        tcli = tcli + [rho, pval]
    ctsli.append(tcli)

[print(x) for x in ctsli]

pd.DataFrame(ctsli, columns=["IN","ConPr", "p ConPr","CC AB","p CC AB","CC BC","p CC BC","CC CA","p CC CA", "BC A","p BC A","BC B", "p BC B", "BC C", "p BC C"]).to_csv("metriccor.csv", index="False")
