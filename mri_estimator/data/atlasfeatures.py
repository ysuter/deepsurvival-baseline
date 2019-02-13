import SimpleITK as sitk
from os import path, walk
import numpy as np
import csv

labeldict = {
  2: "Left-Cerebral-White-Matter",
  3: "Left-Cereral-Cortex",
  4: "Left-Lateral-Ventricle",
  5: "Left-Inf-Lat-Vent",
  7: "Left-Cerebellum-White-Matter",
  8: "Left-Cerebellum-Cortex",
 10: "Left-Thalamus-Proper",
 11: "Left-Caudate",
 12: "Left-Putamen",
 13: "Left-Pallidum",
 14: "3rd-Ventricle",
 15: "4th-Ventricle",
 16: "Brain-Stem",
 17: "Left-Hippocampus",
 18: "Left-Amygdala",
 24: "CSF",
 26: "Left-Accumbens-area",
 28: "Left-VentralDC",
 30: "Left-vessel",
 31: "Left-choroid-plexus",
 41: "Right-Cerebral-White-Matter",
 42: "Right-Cerebral-Cortex",
 43: "Right-Lateral-Ventricle",
 44: "Right-Inf-Lat-Vent",
 46: "Right-Cerebellum-White-Matter",
 47: "Right-Cerebellum-Cortex",
 49: "Right-Thalamus-Proper",
 50: "Right-Caudate",
 51: "Right-Putamen",
 52: "Right-Pallidum",
 53: "Right-Hippocampus",
 54: "Right-Amygdala",
 58: "Right-Accumbens-area",
 60: "Right-VentralDC",
 62: "Right-vessel",
 63: "Right-choroid-plexus",
 72: "5th-Ventricle",
 77: "WM-hypointensities",
 85: "Optic-Chiasm",
251: "CC_Posterior",
252: "CC_Mid_Posterior",
253: "CC_Central",
254: "CC_Mid_Anterior",
255: "CC_Anterior"}

# load atlas label file
atlasLabel = sitk.ReadImage('/usr/local/freesurfer/subjects/fsaverage/mri/aseg.nii.gz')
atlaslabelarr = sitk.GetArrayFromImage(atlasLabel)
atlasLabelInts = np.unique(sitk.GetArrayFromImage(atlasLabel))

# loop over all subjects
rootdir = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/Brats18/Data/Reglabels'
csvpath = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/Brats18/Data/Reglabels/atlasfeat1_ET_NCR.csv'

with open(csvpath, 'w') as file:
    writer = csv.writer(file, delimiter=',')
    row = ['name'] + ['Age']+ ['Survival'] + list(labeldict.values())
    writer.writerow(row)
    file.close()

subjectdirs = [x[0] for x in walk(rootdir)][1:]

featmat = np.zeros([len(subjectdirs), len(labeldict.keys())])



subjlist = [path.split(x)[1] for x in subjectdirs]


for diridx, subjdir in enumerate(subjectdirs):
    print(diridx)

    currentsubj = path.split(subjdir)[1]

    tumor = sitk.ReadImage(path.join(subjdir, currentsubj)+ '_segreg.nii.gz')
    tumorarr = sitk.GetArrayFromImage(tumor)

    for idx, currlabel in enumerate(atlasLabelInts[atlasLabelInts != 0]):

        overlap = atlaslabelarr[(atlaslabelarr == currlabel) & ((tumorarr == 1) | (tumorarr == 4))]
        segsize = np.count_nonzero(atlaslabelarr[atlaslabelarr == currlabel])
        overlapsize = np.count_nonzero(overlap)
        # print(segsize)
        # print(overlapsize)
        if (overlapsize == 0) or (segsize == 0):
            overlapsize = 0
            tumorload = 0
        else:
            tumorload = overlapsize / segsize

        featmat[diridx, idx] = tumorload

subjlist = [path.split(x)[1] for x in subjectdirs]

with open(csvpath, 'a') as file:
    writer = csv.writer(file, delimiter=',')

    for line in range(len(subjectdirs)):
        currage = np.NaN
        currsurv = np.NaN

        csvfile_info = csv.reader(open(
            '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/Brats18/Data/TrainVal/survdata_featlist.csv',
            "rt", encoding='utf-8'), delimiter=",")
        for inforow in csvfile_info:
            # if current rows 2nd value is equal to input, print that row
            if subjlist[line] == inforow[0]:
                currage = inforow[1]
                currsurv = inforow[2]


        row = [subjlist[line]] + [currage] + [currsurv] + featmat[line].tolist()
        writer.writerow(row)
file.close()



    #print(overlap.shape)

