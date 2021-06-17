import os
import subprocess
import glob
import pandas as pd
import statistics as stat

cwd = os.getcwd()

rlog = open("racipe_log.txt", "w")

tpdir = "TOPO710"
rcpdir = "/home/shdw/RACIPE-1.0-master/"
numC = 3

os.chdir(tpdir)
tpfl = glob.glob("*.topo")

os.chdir("..")

rcpli = []
bmcli = []

subprocess.run("mkdir Results", shell=True)
os.chdir("Results")

for t in tpfl:
    subprocess.run("mkdir " + t[:-5], shell=True)
    mkrfol = "mkdir " + t[:-5] + "/1 " + t[:-5] + "/2 " + t[:-5] + "/3"
    subprocess.run(mkrfol, shell=True)
    for sbdr in range(1,4):
        tfc = "cp " + cwd + "/" + tpdir + "/" + t + " " + cwd + "/Results/" + t[:-5] + "/" + str(sbdr)  + "/"
        subprocess.call(tfc, shell=True)
        rcpli.append("./RACIPE " + cwd + "/Results/" + t[:-5] + "/" + str(sbdr) + "/*.topo -num_paras 10 -num_ode 10")
        bcp = "cp " + cwd + "/bmc_cor.py " + cwd + "/Results/" + t[:-5] + "/" + str(sbdr)  + "/"
        subprocess.call(bcp, shell=True)
        bmcli.append("python3 " + cwd + "/Results/" + t[:-5] + "/" + str(sbdr) + "/bmc_cor.py")


Rbtchs = [rcpli[i:i + numC] for i in range(0, len(rcpli), numC)]
os.chdir(rcpdir)
for rr in Rbtchs:
    rnRCP = " & ".join(rr) + " & wait"
    print(rnRCP)
    subprocess.run(rnRCP, shell=True)
    rlog.write(rnRCP + "\n")

Bbtchs = [bmcli[i:i + numC] for i in range(0, len(bmcli), numC)]
os.chdir(cwd+"/Results/")
for br in Bbtchs:
    brn = " & ".join(br) + " & wait"
    subprocess.run(brn, shell=True)

ccl = []
bcl = []

for t in tpfl:
    CCcmd = "cat " + t[:-5] + "/1/cor.txt " + t[:-5] + "/2/cor.txt "+ t[:-5] + "/3/cor.txt " + "> " + t[:-5] + "/CC.txt"
    BCcmd = "cat " + t[:-5] + "/1/bmc.txt " + t[:-5] + "/2/bmc.txt "+ t[:-5] + "/3/bmc.txt " + "> " + t[:-5] + "/BC.txt"
    #GKcmd = "cat " + t[:-5] + "/1/gknorm.txt " + t[:-5] + "/2/gknorm.txt "+ t[:-5] + "/3/gknorm.txt " + "> " + t[:-5] + "/GK.txt"
    subprocess.run(CCcmd, shell=True)
    subprocess.run(BCcmd, shell=True)
    #subprocess.run(GKcmd, shell=True)
    os.chdir(t[:-5])
    cdf = pd.read_csv("CC.txt", header=None)
    ccl.append([t[:-5], cdf[0].mean(), cdf[0].std()])
    bdf = pd.read_csv("BC.txt", header=None)
    bcl.append([t[:-5], bdf[0].mean(), bdf[0].std(), bdf[1].mean(), bdf[1].std()])
    os.chdir(cwd+"/Results/")

Cdf = pd.DataFrame(ccl, columns = ["RN", "Pearson Mean", "Std"])
Bdf = pd.DataFrame(bcl, columns = ["RN", "BC A", "Std(BC A)", "BC B", "Std(BC B)"])

Cdf.to_excel("Corealtion.xlsx")
Bdf.to_excel("Bimodality.xlsx")
