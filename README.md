# Modularity

## Scripts
- __rnd_net.py__ - Generate random networks of given number of nodes and edges with an embedded motif.

- __rmv_dup.py__ - Checks for duplicates in the random networks generated and removes them if found. Generates a log file with the duplicate networks.

- __bmc_cor,py__ - Reads the data created by RACIPE and calculates the bimodality coefficent and correlation coeffiencent between the motifs nodes, writes the values into csv files. 

- __racipe_run.py__ - Creates directories and for each topology file generated by rnd_net.py and simulates each topo file three times, analyses the data using bmc_cor.py and to create csv file of the compiled data of all the 100 netwokrks.


## Folders
### IntroData
Contains data and plot codes for Isolated TS and TT motifs.

### TS_Plots
Contains data and plot codes for TS embedded in random networks.

### TT_Plots
Contains data and plot codes for TT embedded in random networks.


