#!/bin/bash

cd / run / user / 1000 / gvfs / smb - share: server = istb - brain.unibe.ch, share = data / mia / MIALab2017 /

for d in * /; do
# check if current subject was already processed
if [ ! -d../ MIALab2017_dataprep / ${d} /]; then

mkdir - p.. / MIALab2017_dataprep /${d}

## NATIVE SPACE
# extract grey matter labels (2: left cerebral white matter, 41: right cerebral em matter, 7: left cerebellum wm, 46: right cerebellum wm
mri_binarize - -i ${d} / T1w / aparc + aseg.nii.gz - -match
2
41
7
46 - -binval
1 - -o.. / MIALab2017_dataprep /${d} / wm_native.nii.gz

# extract grey matter labels (3: left cerebral grey matter, 42: right cerebral grey matter, 8: left cerebellum cortex, 47: right cerebellum cortex
mri_binarize - -i ${d} / T1w / aparc + aseg.nii.gz - -match
3
42
8
47
1000
1001
1002
1003
1004
1005
1006
1007
1008
1009
1010
1011
1012
1013
1014
1015
1016
1017
1018
1019
1020
1021
1022
1023
1024
1025
1026
1027
1028
1029
1030
1031
1032
1033
1034
2000
2001
2002
2003
2004
2005
2006
2007
2008
2009
2010
2011
2012
2013
2014
2015
2016
2017
2018
2019
2020
2021
2022
2023
2024
2025
2026
2027
2028
2029
2030
2031
2032
2033
2034 - -binval
2 - -o.. / MIALab2017_dataprep /${d} / gm_native.nii.gz

# extract ventricle labels (4: left lat. ventr., 5: left inf. lat. ventr., 14: 3rd ventr., 15: 4th ventr., 43: right lat. ventr., 44: right inf. lat. ventr.,
mri_binarize - -i ${d} / T1w / aparc + aseg.nii.gz - -match
4
5
14
15
43
44
75
76 - -binval
3 - -o.. / MIALab2017_dataprep /${d} / ventr_native.nii.gz

# combine native labels
mri_concat - -i.. / MIALab2017_dataprep /${d} / wm_native.nii.gz - -i.. / MIALab2017_dataprep /${
                                                                                                    d} / gm_native.nii.gz - -i.. / MIALab2017_dataprep /${
                                                                                                                                                             d} / ventr_native.nii.gz - -sum - -o.. / MIALab2017_dataprep /${
                                                                                                                                                                                                                                d} / labels_native.nii.gz

# clean up
rm - f.. / MIALab2017_dataprep /${d} / wm_native.nii.gz - f.. / MIALab2017_dataprep /${
                                                                                          d} / gm_native.nii.gz - f.. / MIALab2017_dataprep /${
                                                                                                                                                  d} / ventr_native.nii.gz

## SAME FOR MNI ATLAS SPACE
# extract grey matter labels (2: left cerebral white matter, 41: right cerebral grey matter, 7: left cerebellum wm, 46: right cerebellum wm
mri_binarize - -i ${d} / MNINonLinear / aparc + aseg.nii.gz - -match
2
41
7
46 - -binval
1 - -o.. / MIALab2017_dataprep /${d} / wm_mniatlas.nii.gz

# extract grey matter labels (3: left cerebral grey matter, 42: right cerebral grey matter, 8: left cerebellum cortex, 47: right cerebellum cortex
mri_binarize - -i ${d} / MNINonLinear / aparc + aseg.nii.gz - -match
3
42
8
47
1000
1001
1002
1003
1004
1005
1006
1007
1008
1009
1010
1011
1012
1013
1014
1015
1016
1017
1018
1019
1020
1021
1022
1023
1024
1025
1026
1027
1028
1029
1030
1031
1032
1033
1034
2000
2001
2002
2003
2004
2005
2006
2007
2008
2009
2010
2011
2012
2013
2014
2015
2016
2017
2018
2019
2020
2021
2022
2023
2024
2025
2026
2027
2028
2029
2030
2031
2032
2033
2034 - -binval
2 - -o.. / MIALab2017_dataprep /${d} / gm_mniatlas.nii.gz

# extract ventricle labels (4: left lat. ventr., 5: left inf. lat. ventr., 14: 3rd ventr., 15: 4th ventr., 43: right lat. ventr., 44: right inf. lat. ventr.,
mri_binarize - -i ${d} / MNINonLinear / aparc + aseg.nii.gz - -match
4
5
14
15
43
44 - -binval
3 - -o.. / MIALab2017_dataprep /${d} / ventr_mniatlas.nii.gz

# combine native labels
mri_concat - -i.. / MIALab2017_dataprep /${d} / wm_mniatlas.nii.gz - -i.. / MIALab2017_dataprep /${
                                                                                                      d} / gm_mniatlas.nii.gz - -i.. / MIALab2017_dataprep /${
                                                                                                                                                                 d} / ventr_mniatlas.nii.gz - -sum - -o.. / MIALab2017_dataprep /${
                                                                                                                                                                                                                                      d} / labels_mniatlas.nii.gz

# clean up
rm - f.. / MIALab2017_dataprep /${d} / wm_mniatlas.nii.gz - f.. / MIALab2017_dataprep /${
                                                                                            d} / gm_mniatlas.nii.gz - f.. / MIALab2017_dataprep /${
                                                                                                                                                      d} / ventr_mniatlas.nii.gz

## Copy to folder: T1 native & atlas, T2 native & atlas, all these bias-field corrected, non-bias-field-corrected, skull-stripped and non-skull-stripped
cp ${d} / T1w / T1w_acpc_dc.nii.gz.. / MIALab2017_dataprep /${d} / T1native.nii.gz
cp ${d} / T1w / T1w_acpc_dc_restore.nii.gz.. / MIALab2017_dataprep /${d} / T1native_biasfieldcorr.nii.gz
cp ${d} / T1w / T1w_acpc_dc_restore_brain.nii.gz.. / MIALab2017_dataprep /${d} / T1native_biasfieldcorr_noskull.nii.gz

cp ${d} / T1w / T2w_acpc_dc.nii.gz.. / MIALab2017_dataprep /${d} / T2native.nii.gz
cp ${d} / T1w / T2w_acpc_dc_restore.nii.gz.. / MIALab2017_dataprep /${d} / T2native_biasfieldcorr.nii.gz
cp ${d} / T1w / T2w_acpc_dc_restore_brain.nii.gz.. / MIALab2017_dataprep /${d} / T2native_biasfieldcorr_noskull.nii.gz

cp ${d} / T1w / BiasField_acpc_dc.nii.gz.. / MIALab2017_dataprep /${d} / BiasFieldnative.nii.gz
cp ${d} / T1w / brainmask_fs.nii.gz.. / MIALab2017_dataprep /${d} / Brainmasknative.nii.gz

# atlas space
cp ${d} / MNINonLinear / T1w.nii.gz.. / MIALab2017_dataprep /${d} / T1mni.nii.gz
cp ${d} / MNINonLinear / T1w_restore
.2.nii.gz.. / MIALab2017_dataprep /${d} / T1mni_biasfieldcorr.nii.gz
cp ${d} / MNINonLinear / T1w_restore_brain.nii.gz.. / MIALab2017_dataprep /${d} / T1mni_biasfieldcorr_noskull.nii.gz

cp ${d} / MNINonLinear / T2w.nii.gz.. / MIALab2017_dataprep /${d} / T2mni.nii.gz
cp ${d} / MNINonLinear / T2w_restore
.2.nii.gz.. / MIALab2017_dataprep /${d} / T2mni_biasfieldcorr.nii.gz
cp ${d} / MNINonLinear / T2w_restore_brain.nii.gz.. / MIALab2017_dataprep /${d} / T2mni_biasfieldcorr_noskull.nii.gz

cp ${d} / MNINonLinear / BiasField.nii.gz.. / MIALab2017_dataprep /${d} / BiasFieldmni.nii.gz
cp ${d} / MNINonLinear / brainmask_fs.nii.gz.. / MIALab2017_dataprep /${d} / Brainmaskmni.nii.gz

echo
"processed $d"
fi
done

/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/Brats18/Data/Reglabels

mri_robust_register --mov /media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/Brats18/Data/TrainVal/Brats18_TCIA01_147_1/Brats18_TCIA01_147_1_t1ce.nii.gz --dst /usr/local/freesurfer/subjects/fsaverage/mri/brain.mgz --lta
testaffine.tfm --iscale --initorient --affine --satit --mapmov aligned.nii.gz