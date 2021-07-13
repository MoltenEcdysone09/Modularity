#For C1
from scipy.stats import skew, kurtosis, pearsonr, zscore, spearmanr
import glob
import pandas as pd
import math
import pathlib
import os
import itertools

cwd = pathlib.Path(__file__).parent.absolute()
os.chdir(cwd)
print(os.getcwd())

slfl = glob.glob(r"*solution*.dat")

#number of nodes in the network
N = 9

#number of embedded nodes in the network (motif)
embN = 4

# Node List to find out the first N nodes (whose postions in the .prs file need to be found)
nlst = "ABCDEFGHIJKLMNOPQRSTUVWZYZ"

# Function to find out the indices of the nodes of the embeded network
def find_col_I(embN):
    #open the .prs file which has the order in which the paramters are listed in all the solution files
    pro = str(glob.glob(r"*.prs")[0])
    # read the file as a dataframe
    prdf = pd.read_csv(pro, sep = "\t")
    # Select the first column of the dataframe
    prli = prdf.iloc[:,0]

    # Empty list to store the column indices
    coli = []

    # Find out all the required node column index in the paramter files
    for x in range(0,embN):
        for l in range(0,len(prli)):
            if "Prod_of_" + nlst[x] in prli[l]:
                coli.append(l)
    return coli

colI = find_col_I(embN)

# Function to g/k normalise the respective columns of the indices
def gkn(paralin, solin, colI, v):
    gknV = []
    for e in colI:
        gk = float(paralin[2+e])/float(paralin[2+e+N])
        nod = pow(2,float(solin[(2 + N*v)]))/gk
        nod = math.log(nod,2)
        gknV.append(nod)
    return gknV

#prfl = str(glob.glob(r"*_parameters.dat")[0])

prf = open(str(glob.glob(r"*_parameters.dat")[0]), "r")
prml = prf.readlines()

gkvdf = []

for n in range(int(len(slfl))):
    fh = open(slfl[n])
    lines = fh.readlines()
    # sln ==> gives the number of the solution file
    sln = int(slfl[n][-5])
    #print(sln)
    if sln == 0:
        sln = 10
    for r in range(len(lines)):
        #print(lines[r])
        #strips the \n at the end of the line
        lines[r] = lines[r].rstrip("\n")
        #strips the \t in the line, outputs a list
        lines[r] = lines[r].split("\t")
        # index of the corresponding line in the parameters.dat file
        #print(lines[r][0])
        l = int(lines[r][0]) - 1
        #print(l)
        #prepare the list of the the diferent parameters
        #print(prml[l])
        prml[l] = prml[l].rstrip("\n")
        prml[l] = prml[l].split("\t")
        #this loop writes in data.txt
        #it divides the line into chunks of N values (i.e. N nodes)
        for v in range(sln):
            #print(v)
            gkvdf.append(gkn(prml[l], lines[r], colI, v))

# create a list of column names
colname = []
for x in range(0,embN):
    colname.append(nlst[x])
# Create a dataframe of the list of list
gvkdf = pd.DataFrame(gkvdf, columns = colname)

for x in range(0,embN):
    gvkdf.iloc[:,x] = zscore(gvkdf.iloc[:,x])

gvkdf.to_csv("gknorm.txt", index = False)

with open("corNP.txt","w") as cof:
    for x, y in itertools.combinations(colname,2):
        corv = spearmanr(gvkdf[x], gvkdf[y])
        cof.write(str(corv[0])+","+str(corv[1])+",")
    cof.write("\n")

def bmc_calc(X):
    gX = skew(X)
    kX = kurtosis(X, fisher=True)
    n = len(X)
    num = (pow((n-1),2))/((n-2)*(n-3))
    bcX = (pow(gX,2) + 1)/(kX + 3*num)
    return bcX

with open("bmc.txt","w") as bmc:
    for x in range(0,embN):
        bmc.write(str(bmc_calc(gvkdf.iloc[:,x]))+",")
    bmc.write("\n")
