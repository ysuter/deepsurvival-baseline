import pandas as pd

# inpcsv = '/run/user/1000/gvfs/smb-share:server=istb-brain,share=data/mia/BrainOncology/BraTS/Brats2018/Features/Testing/shapeOWN_test.csv'
# inpcsv = '/run/user/1000/gvfs/smb-share:server=istb-brain,share=data/mia/BrainOncology/BraTS/Brats2018/Features/shapeOWN_val.csv'
# inpcsv = '/run/user/1000/gvfs/smb-share:server=istb-brain,share=data/mia/BrainOncology/BraTS/Brats2018/Features/shapeOWN_val.csv'
inpcsv = '/home/yannick/Dropbox/Doktorat/BraTS/Testeval/shapeOWN_valnew.csv'

inp = pd.read_csv(inpcsv)

# extract volume columns and calculate fractions
cetvol = inp['volet']
edvol = inp['voled']
ncrvol = inp['volncr']

# cetedratio = cetvol / edvol
# cetncrratio = cetvol / ncrvol
# ncredratio = ncrvol / edvol

cetedratio = pd.DataFrame({'cetedratio': cetvol / edvol})
cetncrratio = pd.DataFrame({'cetncrratio': cetvol / ncrvol})
ncredratio = pd.DataFrame({'ncredratio': ncrvol / edvol})

out = pd.DataFrame(inp['SubjID']).join(cetedratio).join(cetncrratio).join(ncredratio)

outpath = '/home/yannick/Desktop/test/volumeratios_val.csv'
out.to_csv(outpath, index=False)


fullout = inp.join(cetedratio).join(cetncrratio).join(ncredratio)
print(fullout)
outpath_full = '/home/yannick/Dropbox/Doktorat/BraTS/Testeval/shapeOWN_val_volumeratios.csv'
fullout.to_csv(outpath_full, index=False)
print('done')