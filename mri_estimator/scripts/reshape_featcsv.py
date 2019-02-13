#!/home/yannick/miniconda3/envs/tensorflow/bin/python

import csv
import os
import pandas as pd
import numpy as np

#csvpath = '/home/yannick/Dropbox/Doktorat/BraTS/Testeval/radiomicsout_validation_new.csv'
#csvpath = '/hd/BRATS/Brats18/PyRad_Featureextractor/results_missing.csv'
csvpath = '/home/yannick/Dropbox/Doktorat/BraTS/Testeval/results_val_def.csv'
csvpath_out = '/hd/BRATS/Brats18/PyRad_Featureextractor/results_val_reshaped_DEF.csv'
#csvpath_out = '/hd/BRATS/Brats18/PyRad_Featureextractor/results_train_reshaped_DEF.csv'

# inp = pd.read_csv(csvpath)
#
# print(inp)
# test = inp.values
#
# print(inp.shape)

with open(csvpath) as f:
    reader = csv.reader(f)
    #next(reader) # skip header
    data = [r for r in reader]

header = data[0]

numdatalines = int(np.floor(len(data)/12))
print(numdatalines)

numblocks = 12
# new header
newhead = header[0:5]
for idx in range(0,numblocks):
    tmpshape = []
    if idx == 0:
        for blkidx in range(0,3):
            tmpshape = tmpshape + [x + '_' +data[blkidx*numdatalines+1][2] for x in header[6:22]]
    #print(tmpshape)
    #print(len(tmpshape))
    tmp = [x + '_' +data[idx*numdatalines+1][1]+ '_' +data[idx*numdatalines+1][2] for x in header[22:]]
    newhead = newhead + tmpshape + tmp

print(newhead)
#print(data[0])
print(len(newhead))

# reshape blocks
onlydata = np.asarray(data[1:][:])
#print(onlydata.shape)

# reshaped = []
reshaped = onlydata[0:numdatalines,0:6]
for blockidx in range(0,numblocks):
    #print(blockidx)
    if blockidx == 0:
        #reshaped = onlydata[0:numdatalines,0:6]
        #print(reshaped.shape)
        #print(len(reshaped[0]))
        #print(reshaped)
        #print(numdatalines)
        for blkidx in range(0,3):
            print(blkidx)
            #reshaped = np.hstack((reshaped, onlydata[(blkidx*numdatalines*4):(blkidx*numdatalines*4+1)][7:22]))
            print(reshaped.shape)
            print((onlydata[(blkidx * numdatalines):((blkidx+1)*numdatalines)].shape))
            reshaped = np.hstack((reshaped, onlydata[(blkidx * numdatalines):((blkidx+1)*numdatalines), 6:22]))
        reshaped = np.hstack((reshaped, onlydata[(blockidx * numdatalines):((blockidx + 1) * numdatalines), 22:]))
    else:
        print("###############")
        print(reshaped.shape)
        #print(onlydata[(blockidx*numdatalines):((blockidx+1)*numdatalines),22:].shape)
        reshaped = np.hstack((reshaped, onlydata[(blockidx*numdatalines):((blockidx+1)*numdatalines),22:]))


print(reshaped.shape)
print(len(newhead))
outdata = np.vstack((newhead,reshaped))

# remove unused columns
outdata=np.delete(outdata,[1,2], axis=1)

# write to csv
with open(csvpath_out,"w") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(outdata)
exit()
