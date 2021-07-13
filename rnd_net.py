# Script to generate random networks of N nodes and E edges
# Change the N, E, RN variables
# To embed a network, make changes to adjecency matrix from line 78

# Import required packages
import numpy as np
import random
import os

# Total Number of Nodes
N = 9
# Number of Edges (Except the embedded network)
E = 10

# Number of Random Netwroks needed
RN = 3

# Create a directory in current working directory to store topo files
os.mkdir("TOPO"+str(N)+str(E))

# Change current working directory
os.chdir("TOPO"+str(N)+str(E))

# Placeholder letters fro writing in topofile
# First N letters will belong to the embedded nodes
nod_ary = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Make Network function
# Inputs: N - Total no of nodes, E - No. of Edges
# Output: rnet - adjecency matrix
# Exclude certain connection which may change the motif
exE = [[0,3],[1,2],[2,1],[3,0]]
def mk_net(N,E):
    e = 0
    # e will be the row number of the matrix
    # this is done so that each node has atleat one connection
    while e < N:
        # e_cor gives the node with which e chould make a connection
        e_cor = random.randint(0,N-1)
        # e_val gives the nature of the connection
        e_val = random.randint(1,2)
        # check to see if embeded network in no changed
        if e != e_cor and rnet[e,e_cor] == 0 and [e,e_cor] not in exE: 
            rnet[e,e_cor] = e_val
            e = e + 1
    # Once all the nodes have atleast one connection, randomly assign connections.
    e = 0
    while e < E - N:
        # .sample is used as it will not give [x,x] as output, so no self activation
        ## Change if self actiavtion needed in random network nodes
        e_cor = random.sample(range(0,N),2)
        e_val = random.randint(1,2)
        if rnet[e_cor[0], e_cor[1]] == 0 and e_cor not in exE:
            rnet[e_cor[0], e_cor[1]] = e_val
            e = e + 1
    return rnet

# Function to convert adjecency matrix into topo file lines
# Inputs: rnet - adjecency matrix
# Output: tpli - list of lines to be written in topo file
def topo_lin(rnet):
    # Loop through rnet to access each row
    for n in range(0,N):
        # Loop through each element of the row
        for m in range(0,N):
            # If an edge is present then write it into a topofile
            if rnet[n,m] != 0:
                tpli.append(nod_ary[n] + " " + nod_ary[m] + " " + str(rnet[n,m]))

    return tpli

# Generation of random netwroks and writing then into topo filles
for r in range(0,RN):
    # rnet - initilising the empty adjecency matrix
    rnet = np.zeros((N,N), dtype=int)
    # Initiliase array with lines to be written in topo file
    tpli = ["Source Target Type"]

    # Embedding the toggle switch into the adjecency matrix
    rnet[0,1] = 2
    rnet[0,2] = 2
    rnet[1,0] = 2
    rnet[1,3] = 2
    rnet[2,0] = 2
    rnet[2,3] = 2
    rnet[3,1] = 2
    rnet[3,2] = 2
    # Run the make netwrok function; returns rnet
    mk_net(N,E)
    #print(np.count_nonzero(rnet))
    # Run the topo_lin function; returns tpli which has line to be written in the topo file
    topo_lin(rnet)
    #print(len(tpli))
    # Write the elements in tpli into a topo file as lines
    with open('RN' + format(r+1, '03d') + ".topo", 'w') as tf:
        # E+2+1 as 2 nodes in embedded motif and 1 header line
        for l in range(0,len(tpli)):
            tf.write(tpli[l]+"\n")

