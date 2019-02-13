import numpy as np
import os
import SimpleITK as sitk

# thanks to https://github.com/Kamnitsask/deepmedic/blob/master/deepmedic/image/processing.py
def reflectImageArrayIfNeeded(reflectFlags, imageArray):
    stepsForReflectionPerDimension = [-1 if reflectFlags[0] else 1, -1 if reflectFlags[1] else 1,
                                      -1 if reflectFlags[2] else 1]

    reflImageArray = imageArray[::stepsForReflectionPerDimension[0], ::stepsForReflectionPerDimension[1],
                     ::stepsForReflectionPerDimension[2]]


    return reflImageArray

rootdir = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/Brats18/Data/TrainVal_Augment'

subjlist = ['Brats18_TCIA08_167_1','Brats18_TCIA08_242_1','Brats18_TCIA08_319_1','Brats18_TCIA08_469_1','Brats18_TCIA08_218_1','Brats18_TCIA08_406_1','Brats18_TCIA08_280_1','Brats18_TCIA08_105_1','Brats18_TCIA08_278_1','Brats18_TCIA06_247_1','Brats18_TCIA06_372_1','Brats18_TCIA06_165_1','Brats18_TCIA06_409_1','Brats18_TCIA06_184_1','Brats18_TCIA05_277_1','Brats18_TCIA05_478_1','Brats18_TCIA04_437_1','Brats18_TCIA04_361_1','Brats18_TCIA04_192_1','Brats18_TCIA04_479_1','Brats18_TCIA04_111_1','Brats18_TCIA04_343_1','Brats18_TCIA04_149_1','Brats18_TCIA03_474_1','Brats18_TCIA03_419_1','Brats18_TCIA03_199_1','Brats18_TCIA03_133_1','Brats18_TCIA03_296_1','Brats18_TCIA03_257_1','Brats18_TCIA03_498_1','Brats18_TCIA03_138_1','Brats18_TCIA03_338_1','Brats18_TCIA03_265_1','Brats18_TCIA03_375_1','Brats18_TCIA03_121_1','Brats18_TCIA02_274_1','Brats18_TCIA02_473_1','Brats18_TCIA02_322_1','Brats18_TCIA02_179_1','Brats18_TCIA02_368_1','Brats18_TCIA02_135_1','Brats18_TCIA02_471_1','Brats18_TCIA02_394_1','Brats18_TCIA02_300_1','Brats18_TCIA02_151_1','Brats18_TCIA02_118_1','Brats18_TCIA02_226_1','Brats18_TCIA02_455_1','Brats18_TCIA02_283_1','Brats18_TCIA02_430_1','Brats18_TCIA02_321_1','Brats18_TCIA02_314_1','Brats18_TCIA02_290_1','Brats18_TCIA02_377_1','Brats18_TCIA02_198_1','Brats18_TCIA02_331_1','Brats18_TCIA02_491_1','Brats18_TCIA01_150_1','Brats18_TCIA01_335_1','Brats18_TCIA01_411_1','Brats18_TCIA01_203_1','Brats18_TCIA01_231_1','Brats18_TCIA01_390_1','Brats18_TCIA01_235_1','Brats18_TCIA01_499_1','Brats18_TCIA01_412_1','Brats18_TCIA01_448_1','Brats18_TCIA01_401_1','Brats18_TCIA01_147_1','Brats18_TCIA01_378_1','Brats18_TCIA01_201_1','Brats18_TCIA01_429_1','Brats18_TCIA01_186_1','Brats18_TCIA01_460_1','Brats18_TCIA01_190_1','Brats18_TCIA01_425_1','Brats18_2013_11_1','Brats18_2013_27_1','Brats18_CBICA_BHM_1','Brats18_CBICA_BHB_1','Brats18_CBICA_AZH_1','Brats18_CBICA_AZD_1','Brats18_CBICA_AYW_1','Brats18_CBICA_AYU_1','Brats18_CBICA_AYI_1','Brats18_CBICA_AYA_1','Brats18_CBICA_AXW_1','Brats18_CBICA_AXQ_1','Brats18_CBICA_AXO_1','Brats18_CBICA_AXN_1','Brats18_CBICA_AXM_1','Brats18_CBICA_AXL_1','Brats18_CBICA_AXJ_1','Brats18_CBICA_AWI_1','Brats18_CBICA_AWH_1','Brats18_CBICA_AWG_1','Brats18_CBICA_AVV_1','Brats18_CBICA_AVJ_1','Brats18_CBICA_AVG_1','Brats18_CBICA_AUR_1','Brats18_CBICA_AUQ_1','Brats18_CBICA_AUN_1','Brats18_CBICA_ATX_1','Brats18_CBICA_ATV_1','Brats18_CBICA_ATP_1','Brats18_CBICA_ATF_1','Brats18_CBICA_ATD_1','Brats18_CBICA_ATB_1','Brats18_CBICA_ASY_1','Brats18_CBICA_ASW_1','Brats18_CBICA_ASV_1','Brats18_CBICA_ASU_1','Brats18_CBICA_ASO_1','Brats18_CBICA_ASN_1','Brats18_CBICA_ASK_1','Brats18_CBICA_ASH_1','Brats18_CBICA_ASG_1','Brats18_CBICA_ASE_1','Brats18_CBICA_ASA_1','Brats18_CBICA_ARZ_1','Brats18_CBICA_ARW_1','Brats18_CBICA_ARF_1','Brats18_CBICA_AQZ_1','Brats18_CBICA_AQY_1','Brats18_CBICA_AQV_1','Brats18_CBICA_AQU_1','Brats18_CBICA_AQT_1','Brats18_CBICA_AQR_1','Brats18_CBICA_AQQ_1','Brats18_CBICA_AQP_1','Brats18_CBICA_AQO_1','Brats18_CBICA_AQN_1','Brats18_CBICA_AQJ_1','Brats18_CBICA_AQG_1','Brats18_CBICA_AQD_1','Brats18_CBICA_AQA_1','Brats18_CBICA_APZ_1','Brats18_CBICA_APY_1','Brats18_CBICA_APR_1','Brats18_CBICA_AOZ_1','Brats18_CBICA_AOP_1','Brats18_CBICA_AOO_1','Brats18_CBICA_AOH_1','Brats18_CBICA_AOD_1','Brats18_CBICA_ANZ_1','Brats18_CBICA_ANP_1','Brats18_CBICA_ANI_1','Brats18_CBICA_ANG_1','Brats18_CBICA_AMH_1','Brats18_CBICA_AME_1','Brats18_CBICA_ALX_1','Brats18_CBICA_ALU_1','Brats18_CBICA_ALN_1','Brats18_CBICA_ABY_1','Brats18_CBICA_ABO_1','Brats18_CBICA_ABN_1','Brats18_CBICA_ABM_1','Brats18_CBICA_ABE_1','Brats18_CBICA_ABB_1','Brats18_CBICA_AAP_1','Brats18_CBICA_AAL_1','Brats18_CBICA_AAG_1','Brats18_CBICA_AAB_1']
# loop over all folders
# dirlist = [x[0] for x in os.walk(rootdir)]
dirlist = [os.path.join(rootdir, x) for x in subjlist]
#print(dirlist[1:]) # first entry is root folder
#dirlist = dirlist[1:]
print(len(dirlist))
print(os.path.split(dirlist[0]))

for idx, currentdir in enumerate(dirlist):
    # make directory for flipped version
    flippedir = os.path.join(rootdir, currentdir)+'_f'
    os.makedirs(flippedir, exist_ok=True)

    files = [x[2] for x in os.walk(currentdir)][0]

    for fileidx, currfile in enumerate(files):
        inp = sitk.ReadImage(os.path.join(rootdir, currentdir, currfile))
        flipimgarr3 = reflectImageArrayIfNeeded([0, 0, 1], sitk.GetArrayFromImage(inp))
        out = sitk.GetImageFromArray(flipimgarr3)
        out.CopyInformation(inp)
        sitk.WriteImage(out, os.path.join(rootdir, flippedir, currfile))



