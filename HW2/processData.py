# Read Data from a text file

with open('boston.txt') as f:
    lines = f.readlines()

f.close()

# create array of strings

data = []

# create a loop that skips over every other element

for i in range(0, len(lines), 2):
    # remove new line at end of lines[i] & lines[i+1]

    lines[i] = lines[i].rstrip()
    lines[i+1] = lines[i+1].rstrip()

    newLine = lines[i] + ' ' + lines[i+1]
    
    data.append(newLine.strip().split())
    
# convert data to a csv

import csv

with open('boston.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(data)

f.close()