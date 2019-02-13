import os
import glob
import typing as t

import miapy.data as data


class Brats18Collector:

    def __init__(self, root_dir: str, has_grade=False, testing=False) -> None:
        if root_dir.endswith('/'):
            root_dir = root_dir[:-1]
        self.root_dir = root_dir
        self.has_grade = has_grade
        self.testing = testing
        self.subject_files = []
        self._collect()

    def get_subject_files(self) -> t.List[data.SubjectFile]:
        return self.subject_files

    def _collect(self):
        self.subject_files.clear()

        subject_dirs = glob.glob(self.root_dir + ('/*/*' if self.has_grade else '/*'))
        # collect the files
        for subject_dir in subject_dirs:
            if not os.path.isdir(subject_dir):
                continue
            subject = os.path.basename(subject_dir)

            image_files = {}
            label_files = {}
            subject_files = glob.glob(subject_dir + '/*.nii.gz')
            for subject_file in subject_files:
                if subject_file.endswith('flair.nii.gz'):
                    image_files['flair'] = subject_file
                elif subject_file.endswith('t1.nii.gz'):
                    image_files['t1'] = subject_file
                elif subject_file.endswith('t1ce.nii.gz'):
                    image_files['t1c'] = subject_file
                elif subject_file.endswith('t2.nii.gz'):
                    image_files['t2'] = subject_file
                elif subject_file.endswith('seg.nii.gz'):
                    label_files['gt'] = subject_file

            if len(image_files) != 4:
                raise ValueError('did not collect all image files')
            if not self.testing and len(label_files) == 0:
                raise ValueError('did not collect label files')
            csv_file = os.path.join(self.root_dir, 'features_niftynetseg_reshaped_inclshape.csv')
            if not os.path.exists(csv_file):
                raise ValueError('file "{}" missing'.format(csv_file))

            age = {'age': csv_file}
            print(age)
            resectionstatus = {'resectionstatus': csv_file}

            # # add more radiomics features
            # logs1_glszm_LALGLE_T2_1 = {'log-sigma-1-0-mm-3D_glszm_LargeAreaLowGrayLevelEmphasis_T2_l1nectrotic': csv_file}
            # logs1_gldm_SDHGLE_T2_1 = {'log-sigma-1-0-mm-3D_gldm_SmallDependenceHighGrayLevelEmphasis_T2_l1nectrotic': csv_file}
            # logs1_glszm_LALGLE_T2_4 = {'log-sigma-1-0-mm-3D_glszm_LargeAreaLowGrayLevelEmphasis_T2_l4ce': csv_file}
            # logs1_glszm_LALGLE_T2_2 = {'log-sigma-1-0-mm-3D_glszm_LargeAreaLowGrayLevelEmphasis_T2_l2edema': csv_file}
            # logs1_FOR_T2_2 = {'log-sigma-1-0-mm-3D_firstorder_Range_T2_l2edema': csv_file}
            # logs1_gldm_SDHGLE_T2_4 = {'log-sigma-1-0-mm-3D_gldm_SmallDependenceHighGrayLevelEmphasis_T2_l4ce': csv_file}
            # logs1_gldm_SDHGLE_T2_2 = {'log-sigma-1-0-mm-3D_gldm_SmallDependenceHighGrayLevelEmphasis_T2_l2edema': csv_file}
            # logs1_FOR_T2_4 = {'log-sigma-1-0-mm-3D_firstorder_Range_T2_l4ce': csv_file}
            # logs1_FOR_T2_1 = {'log-sigma-1-0-mm-3D_firstorder_Range_T2_l1nectrotic': csv_file}
            # orig_glcm_SE_T1_2 = {'original_glcm_SumEntropy_T1_l2edema': csv_file}
            # orig_glcm_SE_T1_1 = {'original_glcm_SumEntropy_T1_l1nectrotic': csv_file}
            # orig_glcm_SE_T1_4 = {'original_glcm_SumEntropy_T1_l4ce': csv_file}

            # shape features and volumes
            #volncr = {'volncr': csv_file}
            #voled = {'voled': csv_file}
            # volet = {'volet': csv_file}
            # etrimwidth = {'etrimwidth': csv_file}
            # etgeomhet = {'etgeomhet': csv_file}
            # rim_q1_clipped = {'rim_q1_clipped': csv_file}
            # rim_q2_clipped = {'rim_q2_clipped': csv_file}
            # rim_q3_clipped = {'rim_q3_clipped': csv_file}

            survival = {}
            # survclass = {}
            if not self.testing:
                survival['Survival'] = csv_file
                # survclass['survclass'] = csv_file

                # print(survival)
                # if survival < (10*dpm): survclass = 0
                # if ((survival >= 10 * dpm) & (survival <= 15 * dpm)): survclass = 1
                # else: survclass = 2

            sf = data.SubjectFile(subject, images=image_files, labels=label_files, age=age,
                                  #resectionstatus=resectionstatus,
                                  # volncr=volncr, voled=voled, volet=volet,
                                  # etrimwidth=etrimwidth, etgeomhet=etgeomhet, rim_q1_clipped=rim_q1_clipped,
                                  # rim_q2_clipped=rim_q2_clipped, rim_q3_clipped=rim_q3_clipped,
                                  # logs1_glszm_LALGLE_T2_1=logs1_glszm_LALGLE_T2_1,
                                  # logs1_gldm_SDHGLE_T2_1=logs1_gldm_SDHGLE_T2_1,
                                  # logs1_glszm_LALGLE_T2_4=logs1_glszm_LALGLE_T2_4,
                                  # logs1_glszm_LALGLE_T2_2=logs1_glszm_LALGLE_T2_2, logs1_FOR_T2_2=logs1_FOR_T2_2,
                                  #logs1_gldm_SDHGLE_T2_4=logs1_gldm_SDHGLE_T2_4,
                                  # logs1_gldm_SDHGLE_T2_2=logs1_gldm_SDHGLE_T2_2,
                                  # logs1_FOR_T2_4=logs1_FOR_T2_4, logs1_FOR_T2_1=logs1_FOR_T2_1,
                                  # orig_glcm_SE_T1_2=orig_glcm_SE_T1_2, orig_glcm_SE_T1_1=orig_glcm_SE_T1_1,
                                  # orig_glcm_SE_T1_4=orig_glcm_SE_T1_4,
                                  survival=survival)
                                  #survclass=survclass)

            self.subject_files.append(sf)
