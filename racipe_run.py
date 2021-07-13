#General - needs only the number of embedded nodes
import os
import subprocess
import glob
import pandas as pd
import statistics as stat
import itertools
from math import comb

# Number of embedded nodes
embN = 4

#Node names
nme = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Initialise an empty list for column names
cln = []
for x in range(0,embN):
    cln.append(nme[x])

cwd = os.getcwd()

rlog = open("racipe_log.txt", "w")

tpdir = "TOPO910"
rcpdir = "/home/shdw/RACIPE-1.0-master/"
numC = 39

os.chdir(tpdir)
tpfl = sorted(glob.glob("*.topo"))

os.chdir("..")

rcpli = []
bmcli = []

#subprocess.run("mkdir Results", shell=True)
os.chdir("Results")

for t in tpfl:
    #subprocess.run("mkdir " + t[:-5], shell=True)
    #mkrfol = "mkdir " + t[:-5] + "/1 " + t[:-5] + "/2 " + t[:-5] + "/3"
    #subprocess.run(mkrfol, shell=True)
    for sbdr in range(1,4):
        #tfc = "cp " + cwd + "/" + tpdir + "/" + t + " " + cwd + "/Results/" + t[:-5] + "/" + str(sbdr)  + "/"
        #subprocess.call(tfc, shell=True)
        #rcpli.append("./RACIPE " + cwd + "/Results/" + t[:-5] + "/" + str(sbdr) + "/*.topo -num_paras 10 -num_ode 10")
        bcp = "cp " + cwd + "/bmc_cor.py " + cwd + "/Results/" + t[:-5] + "/" + str(sbdr)  + "/"
        subprocess.call(bcp, shell=True)
        bmcli.append("python3 " + cwd + "/Results/" + t[:-5] + "/" + str(sbdr) + "/bmc_cor.py")


#Rbtchs = [rcpli[i:i + numC] for i in range(0, len(rcpli), numC)]
#os.chdir(rcpdir)
#for rr in Rbtchs:
#    rnRCP = " & ".join(rr) + " & wait"
#    print(rnRCP)
#    subprocess.run(rnRCP, shell=True)
#    rlog.write(rnRCP + "\n")

Bbtchs = [bmcli[i:i + numC] for i in range(0, len(bmcli), numC)]
os.chdir(cwd+"/Results/")
for br in Bbtchs:
    brn = " & ".join(br) + " & wait"
    subprocess.run(brn, shell=True)

ccl = []
bcl = []

for t in tpfl:
    CCcmd = "cat " + t[:-5] + "/1/corNP.txt " + t[:-5] + "/2/corNP.txt "+ t[:-5] + "/3/corNP.txt " + "> " + t[:-5] + "/CCNP.txt"
    BCcmd = "cat " + t[:-5] + "/1/bmc.txt " + t[:-5] + "/2/bmc.txt "+ t[:-5] + "/3/bmc.txt " + "> " + t[:-5] + "/BC.txt"
#    GKcmd = "cat " + t[:-5] + "/1/gknorm.txt " + t[:-5] + "/2/gknorm.txt "+ t[:-5] + "/3/gknorm.txt " + "> " + t[:-5] + "/GK.txt"
    subprocess.run(CCcmd, shell=True)
    subprocess.run(BCcmd, shell=True)
#    subprocess.run(GKcmd, shell=True)
    os.chdir(t[:-5])
    cdf = pd.read_csv("CCNP.txt", header=None)
    tmpccl = [t[:-5]]
    for e in range(0,comb(embN,2)*2,2):
        tmpccl = tmpccl + [cdf[e].mean(), cdf[e].std()]
    ccl.append(tmpccl)
    bdf = pd.read_csv("BC.txt", header=None)
    tmpbcl = [t[:-5]]
    for w in range(0,embN):
        tmpbcl = tmpbcl + [bdf[w].mean(), bdf[w].std()]
    bcl.append(tmpbcl)
    os.chdir(cwd+"/Results/")

def CC_colnm(cln):
    cclnm = ["RN"]
    for x, y in itertools.combinations(cln,2):
        cclnm.append("CC "+x+y)
        cclnm.append("STD "+x+y)
    return cclnm

bclnm = ["RN"]
for x in cln:
    bclnm.append("BC "+x)
    bclnm.append("STD "+x)

Cdf = pd.DataFrame(ccl, columns = CC_colnm(cln))
Bdf = pd.DataFrame(bcl, columns = bclnm)

Cdf.to_excel("CorrealtionNP.xlsx")
Cdf.to_csv("CorrelationNP.csv")
Bdf.to_excel("Bimodality.xlsx")
Bdf.to_csv("Bimodality.csv")
