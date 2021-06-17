from scipy.stats import skew, kurtosis, pearsonr, zscore
import glob
import pandas as pd
import math
import pathlib
import os

cwd = pathlib.Path(__file__).parent.absolute()
os.chdir(cwd)
print(os.getcwd())

slfl = glob.glob(r"*solution*.dat")

#prfl = str(glob.glob(r"*_parameters.dat")[0])

prf = open(str(glob.glob(r"*_parameters.dat")[0]), "r")
prml = prf.readlines()

A = []
B = []

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
        l = int(lines[r][0]) - 1
        #prepare the list of the the diferent parameters
        #print(prml[l])
        prml[l] = prml[l].rstrip("\n")
        prml[l] = prml[l].split("\t")
        #this loop writes in data.txt
        #it divides the line into chunks of 4 values (i.e. 4 nodes)
        for v in range(sln):
            #print(v)
            #find the g/k values of the corresponding nodes
            gvkA = float(prml[l][2])/float(prml[l][9])
            gvkB = float(prml[l][3])/float(prml[l][10])
            #nomalise the final values in solution.dat with the g/k values if the corresponding node
            nodA = pow(2,float(lines[r][(2 + 4*v)]))/gvkA
            nodB = pow(2,float(lines[r][(3 + 4*v)]))/gvkB
            #Take log base2 again
            nodA = math.log(nodA,2)
            nodB = math.log(nodB,2)
            #write them down in the dagvk.txt
            #fo.write(str(nodA) + "," + str(nodB) + "\n")
            A.append(nodA)
            B.append(nodB)

A = zscore(A)
B = zscore(B)

with open("gknorm.txt","w") as fo:
    for z in range(0, len(A)):
        fo.write(str(A[z])+","+str(B[z])+"\n")
        #print(str(A[z])+","+str(B[z]))

with open("cor.txt","w") as cof:
    pcor = pearsonr(A,B)
    cof.write(str(pcor[0])+","+str(pcor[1])+"\n")

def bmc_calc(A):
    gA = skew(A)
    kA = kurtosis(A, fisher=True)
    n = len(A)
    num = (pow((n-1),2))/((n-2)*(n-3))
    bcA = (pow(gA,2) + 1)/(kA + 3*num)
    return bcA

with open("bmc.txt","w") as bmc:
    bmc.write(str(bmc_calc(A)) + "," + str(bmc_calc(B)) + "\n")

#print(prms.iloc[:,9])
#print(prms.iloc[:,2])
#print(prms.iloc[:,2]/prms.iloc[:,2+7])

