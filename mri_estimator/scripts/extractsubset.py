import csv
import pandas as pd
import argparse

inpcsv = pd.read_csv('/run/user/1000/gvfs/smb-share:server=istb-brain,share=data/mia/BrainOncology/BraTS/Brats2018/Features/pyradiomics_corrected/pyrad_train_int_features.csv')

subset = inpcsv[['SubjID','Age', 'log-sigma-1-0-mm-3D_glszm_SmallAreaLowGrayLevelEmphasis_Flair_l1nectrotic',
                'log-sigma-1-0-mm-3D_glcm_Imc1_T1_l1nectrotic',
                'log-sigma-1-0-mm-3D_firstorder_Skewness_T1_l1nectrotic',
                'log-sigma-1-0-mm-3D_glrlm_GrayLevelNonUniformity_Flair_l1nectrotic',
                'log-sigma-1-0-mm-3D_glcm_InverseVariance_T1_l2edema',
                'log-sigma-1-0-mm-3D_gldm_DependenceVariance_Flair_l1nectrotic',
                'log-sigma-1-0-mm-3D_firstorder_90Percentile_T1c_l4ce',
                'log-sigma-1-0-mm-3D_glrlm_GrayLevelNonUniformity_T2_l1nectrotic',
                'log-sigma-1-0-mm-3D_glcm_ClusterShade_T2_l4ce',
                'log-sigma-1-0-mm-3D_glcm_MaximumProbability_T1c_l4ce',
                'log-sigma-1-0-mm-3D_glcm_JointEntropy_T1c_l4ce',
                'log-sigma-1-0-mm-3D_glcm_DifferenceVariance_T1c_l1nectrotic',
                'log-sigma-1-0-mm-3D_firstorder_Skewness_T2_l1nectrotic',
                'log-sigma-1-0-mm-3D_gldm_DependenceEntropy_T1_l1nectrotic',
                'log-sigma-1-0-mm-3D_glszm_ZoneEntropy_T1_l1nectrotic',
                'log-sigma-1-0-mm-3D_firstorder_Median_T2_l1nectrotic',
                'log-sigma-1-0-mm-3D_glcm_MaximumProbability_T2_l2edema',
                'log-sigma-1-0-mm-3D_ngtdm_Complexity_T2_l4ce',
                'log-sigma-1-0-mm-3D_glcm_Autocorrelation_Flair_l1nectrotic',
                'log-sigma-1-0-mm-3D_firstorder_Skewness_T2_l4ce',
                'log-sigma-1-0-mm-3D_firstorder_Maximum_Flair_l4ce',
                'log-sigma-1-0-mm-3D_glcm_MaximumProbability_T1_l4ce',
                'log-sigma-1-0-mm-3D_glszm_SmallAreaEmphasis_Flair_l1nectrotic',
                'log-sigma-1-0-mm-3D_glszm_ZoneVariance_T1_l2edema',
                'log-sigma-1-0-mm-3D_glrlm_HighGrayLevelRunEmphasis_Flair_l1nectrotic',
                'log-sigma-1-0-mm-3D_gldm_LargeDependenceHighGrayLevelEmphasis_T2_l4ce',
                'log-sigma-1-0-mm-3D_ngtdm_Strength_T2_l1nectrotic',
                'log-sigma-1-0-mm-3D_gldm_LargeDependenceHighGrayLevelEmphasis_T2_l1nectrotic',
                'log-sigma-1-0-mm-3D_gldm_LargeDependenceHighGrayLevelEmphasis_T1_l1nectrotic',
                'log-sigma-1-0-mm-3D_ngtdm_Busyness_T1_l4ce',
                'log-sigma-1-0-mm-3D_firstorder_InterquartileRange_T1_l4ce',
                'log-sigma-1-0-mm-3D_gldm_LargeDependenceHighGrayLevelEmphasis_T1_l2edema',
                'log-sigma-1-0-mm-3D_firstorder_Range_T2_l1nectrotic',
                'log-sigma-1-0-mm-3D_glszm_LargeAreaHighGrayLevelEmphasis_T1_l1nectrotic',
                'log-sigma-1-0-mm-3D_firstorder_Mean_Flair_l1nectrotic',
                'log-sigma-1-0-mm-3D_glrlm_RunLengthNonUniformityNormalized_T1_l1nectrotic',
                'log-sigma-1-0-mm-3D_glszm_LargeAreaLowGrayLevelEmphasis_T1_l2edema',
                'log-sigma-1-0-mm-3D_glszm_LargeAreaLowGrayLevelEmphasis_T1c_l2edema',
                'log-sigma-1-0-mm-3D_glszm_LargeAreaEmphasis_T1c_l2edema']]

subset.to_csv('/run/user/1000/gvfs/smb-share:server=istb-brain,share=data/mia/BrainOncology/BraTS/Brats2018/Features/pyradiomics_corrected/pyrad_train_int_features_top40.csv', sep=',', index=False)