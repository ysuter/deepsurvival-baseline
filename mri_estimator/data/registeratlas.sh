#!/bin/bash

export FREESURFER_HOME=/usr/local/freesurfer
source /usr/local/freesurfer/SetUpFreeSurfer.sh

# cd /media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/Brats18/Data/TrainVal
cd /run/user/1000/gvfs/smb-share:server=istb-brain,share=data/mia/BrainOncology/BraTS/Brats2018/Testing_seg/Brats18_redoreg2
# cd /run/user/1000/gvfs/smb-share:server=istb-brain,share=data/mia/BrainOncology/BraTS/Brats2018/Testing_seg/niftynet_segmentations_missing

ls -d */ | sed 's#/##'
for d in */; do
    # check if current subject was already processed
    if [ ! -d ../Reglabels/${d}/ ]; then
		mkdir -p ../Reglabels/${d}
	fi

    echo d


    mri_robust_register --mov ${d}${d%/}_t1ce.nii.gz --dst /usr/local/freesurfer/subjects/fsaverage/mri/brain.mgz --lta ../Reglabels/${d}${d%/}_affine.lta --iscale --initorient --affine --satit --maxit 100
    mri_convert --resample_type nearest --apply_transform ../Reglabels/${d}${d%/}_affine.lta ../niftynet_segmentations_missing2/${d%/}.nii.gz ../Reglabels/${d}${d%/}_segreg.nii.gz

    echo "processed $d"

done

echo "done :-)"
