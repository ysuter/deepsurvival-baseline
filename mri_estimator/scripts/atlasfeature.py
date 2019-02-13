import numpy as np
import SimpleITK as sitk
import nibabel as nib
import csv
import os
import miapy.filtering.filter as fltr

atlasT1 = '/usr/local/freesurfer/subjects/fsaverage/mri/brain.nii.gz'
atlasSeg = '/usr/local/freesurfer/subjects/fsaverage/mri/aseg.nii.gz'
nifitylabelroot = '/home/yannick/remoteubelix/BraTS2018/Validation/niftynet_segmentations'
imgroot = '/home/yannick/remoteubelix/BraTS2018/Validation/Brats18Validation_onlySurvival'

inputcsv = '/home/yannick/Dropbox/Doktorat/BraTS/Testeval/survival_evaluation.csv'

def command_iteration(method):

    if (method.GetOptimizerIteration() == 0):
        print("Estimated Scales: ", method.GetOptimizerScales())
    print("{0:3} = {1:7.5f} : {2}".format(method.GetOptimizerIteration(), method.GetMetricValue(),method.GetOptimizerPosition()))

with open(inputcsv, 'r') as readFile:
    reader = csv.reader(readFile)
    lines = list(reader)
# load atlas image
atlasimg = sitk.ReadImage(atlasT1)
# load segmentation labels
atlasSeg = sitk.ReadImage(atlasSeg)

SegArr = sitk.GetArrayFromImage(atlasSeg)
#print(SegArr)
#print(np.unique(SegArr.flatten()))

for lineidx, currentline in enumerate(lines[1:]):
    currentsubj = currentline[0]
    print(currentline)

    niftylabelpath = os.path.join(nifitylabelroot, currentsubj + '.nii.gz')
    subjimgpath = os.path.join(imgroot, currentsubj) + '/' +currentsubj +'_t1.nii.gz'

    niftylabel = sitk.ReadImage(niftylabelpath)
    subjimg = sitk.ReadImage(subjimgpath)

    # register

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep = 1e-4, numberOfIterations = 500, gradientMagnitudeTolerance = 1e-8 )
    R.SetOptimizerScalesFromIndexShift()
    print(atlasimg.GetDimension())
    tx = sitk.CenteredTransformInitializer(atlasimg, subjimg, sitk.AffineTransform())

    R.SetInitialTransform(tx)

    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    outTx = R.Execute(atlasimg, subjimg)

    exit()
    inittfm = sitk.AffineTransform(3)
    tfminitializer = sitk.CenteredTransformInitializerFilter()
    tfminitializer.GEOMETRY

    initial_transform = sitk.CenteredTransformInitializer(atlasT1,
                                                      subjimg,
                                                      inittfm,
                                                      tfminitializer)
    final_transform = sitk.multires_registration(atlasT1, subjimg, initial_transform)

exit()
# #
# 0.   2.   3.   4.   5.   7.   8.  10.  11.  12.  13.  14.  15.  16.
#   17.  18.  24.  26.  28.  30.  31.  41.  42.  43.  44.  46.  47.  49.
#   50.  51.  52.  53.  54.  58.  60.  62.  63.  77.  85. 251. 252. 253.
#  254. 255.
#
# 2 Left-WM
# 3 Left-GM
# 4 Left-Lateral-Ventricle
# 5 Left-Inf-Lat-Vent
# 7 Left-Cerebellum-White-Matter
# 8 Left-Cerebellum-Cortex
# 10 Left-Thalamus-Proper
# 11 Left-Caudate
# 12 Left-Putamen
# 13 Left-Pallidum
# 14 3rd-Ventricle
# 15 4th-Ventricle
# 16 Brain-Stem
# 17 Left-Hippocampus
# 18 Left-Amygdala
#
# 24 CSF
# 26 Left-Accumbens-area
# 28 Left-VentralDC
# 30 Left-vessel
# 31 Left-choroid-plexus
# 41 Right-WM
# 42 Right-GM
# 43 Right-Lateral-Ventricle
# 44 Right-Inf-Lat-Vent
# 46 Right-Cerebellum-White-Matter
# 47 Right-Cerebellum-Cortex
# 49 Right-Thalamus-Proper
# 50 Right-Caudate
# 51 Right-Putamen
# 52 Right-Pallidum
# 53 Right-Hippocampus
# 54 Right-Amygdala
# 58 Right-Accumbens-area
# 60 Right-VentralDC
# 62 Right-vessel
# 63 Right-choroid-plexus
#
# 85 Optic-Chiasm
#
# 251 CC_Posterior
# 252 CC_Mid_Posterior
# 253 CC_Central
# 254 CC_Mid_Anterior
# 255 CC_Anterior
#
