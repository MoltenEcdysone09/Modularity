from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from statannotations.Annotator import Annotator
from math import log
import itertools

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
    if dfl == "810":
        df["Mean Connectivity"] = ["5N & E:2N"]*100
    elif dfl == "2340":
        df["Mean Connectivity"] = ["20N & E:2N"]*100
    elif dfl in ["1320", "1830"]:
        df["Mean Connectivity"] = ["E:2N"] * 100
    elif dfl in ["820", "1340", "1860", "2380"]:
        df["Mean Connectivity"] = ["E:4N"] * 100
    else:
        df["Mean Connectivity"] = ["E:6N"] * 100
    mdf = mdf.append(df)


mdf["In Degree of TT"] = mdf["InA"] + mdf["InB"] + mdf["InC"]

print(mdf.shape)
print(mdf)
print(mdf.shape)

print("b/w 5N & E:2N & 20N & E:2N")
print("CC AB")
print(stats.mannwhitneyu(mdf[(mdf["Mean Connectivity"] == "5N & E:2N")]["CC AB"], mdf[(mdf["Mean Connectivity"] == "20N & E:2N")]["CC AB"]))
print("CC BC")
print(stats.mannwhitneyu(mdf[(mdf["Mean Connectivity"] == "5N & E:2N")]["CC BC"], mdf[(mdf["Mean Connectivity"] == "20N & E:2N")]["CC BC"]))
print("CC AC")
print(stats.mannwhitneyu(mdf[(mdf["Mean Connectivity"] == "5N & E:2N")]["CC AC"], mdf[(mdf["Mean Connectivity"] == "20N & E:2N")]["CC AC"]))
print("BC A")
print(stats.mannwhitneyu(mdf[(mdf["Mean Connectivity"] == "5N & E:2N")]["BC A"], mdf[(mdf["Mean Connectivity"] == "20N & E:2N")]["BC A"]))
print("BC B")
print(stats.mannwhitneyu(mdf[(mdf["Mean Connectivity"] == "5N & E:2N")]["BC B"], mdf[(mdf["Mean Connectivity"] == "20N & E:2N")]["BC B"]))
print("BC C")
print(stats.mannwhitneyu(mdf[(mdf["Mean Connectivity"] == "5N & E:2N")]["BC C"], mdf[(mdf["Mean Connectivity"] == "20N & E:2N")]["BC C"]))
