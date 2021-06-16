# Script to check for duplicate networks in the generated topo files.
# Uses MD5 checksum to filter out duplicate networks. Because of the network generation algorith uses adjecency matrices to geenrate networks and interprets them in the same way for all the networks, if a duplicate network is created , the information in the topo file will be exaclty the same.
# This protperty is used to read the topo files and generate MD5 has values fro the data they contain, if the data is the same, the hash values are also the same, and these duplicates are then identified and removed. A log file is also generated with the all names of the removed files.
# Commend out os.remove() line id deletion is not desired.

# Importing required libraries
import os
import hashlib
import collections

# Changing CWD to the topo file directory (Change is name different)
os.chdir("TOPO710")

# Create a lsit of all the topo files in a directory. Sorting not necessary but added for extra bling.
tpfl = sorted(os.listdir())

# Create an empty list to store the topo file hash codes
tpfh = []

# Create an empty list to store duplicate file names
dupl = []

# Read the topo files and then create thier MD5 check sums
for t in tpfl:
    # Opening the topo file to read, "rb" is used as it will read it diretly as bytes and not unicode
    with open(t, "rb") as hsf:
        # Read teh contents of topo file
        rnt = hsf.read()
        # Create a MD5 hash value for the contents i.e. network
        fmd5 = hashlib.md5(rnt).hexdigest()
        # Check if the has value is already present in the list of hash values
        if fmd5 in tpfh:
            # If present append the file name into dupl
            dupl.append(t)
        else:
            # If not present append the hash value into tpfh 
            tpfh.append(fmd5)

# Open a log file to wite down the deleted topo files and delete them.
with open("duplicate_log.txt","w") as log:
    for d in dupl:
        log.write(d+"\n")
        os.remove(d)
