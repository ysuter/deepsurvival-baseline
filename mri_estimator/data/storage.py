import typing
import re
import os
import csv

import SimpleITK as sitk
import numpy as np

import miapy.data as miapy_data
import miapy.data.conversion as miapy_conv
import miapy.data.creation as miapy_crt
import miapy.data.extraction as miapy_extr
import miapy.data.indexexpression as miapy_expr
import miapy.data.transformation as miapy_tfm
import miapy.data.creation.fileloader as miapy_fileload
import miapy.data.creation.writer as miapy_writer
import miapy.data.creation.callback as miapy_callback

STORE_META_MORPHOMETRY_COLUMNS = 'meta/MORPHOMETRY_COLUMNS'
# STORE_META_INTENSITY_RESCALE_MAX = 'meta/INTENSITY_RESCALE_MAX'
# STORE_IMAGES = 'images'
STORE_MORPHOMETRICS = 'morphometrics'
# STORE_META_CENTROID_TRANSFORM = 'meta/transform/centroids'
# STORE_DEMOGRAPHIC_AGE = 'demographics/age'
# STORE_DEMOGRAPHIC_SEX = 'demographics/sex'


class SurvivalDataLoader(miapy_fileload.Load):

    def __init__(self):
        self.csv_file = None
        self.csv_content = None

    def __call__(self, file_path: str, id_: str, category: str, subject_id: str
                 ) -> typing.Tuple[np.ndarray, typing.Union[miapy_conv.ImageProperties, None]]:
        if category == 'images':
            img = sitk.ReadImage(file_path)
            return sitk.GetArrayFromImage(img), miapy_conv.ImageProperties(img)

        if category == 'labels':
            img = sitk.ReadImage(file_path, sitk.sitkUInt8)
            return sitk.GetArrayFromImage(img), miapy_conv.ImageProperties(img)

        if category == 'age' or category == 'resectionstatus' or category == 'survival':
            if self.csv_file is None or file_path != self.csv_file:
                self.csv_file = file_path
                self.csv_content = {}
                with open(self.csv_file) as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    for row in reader:
                        subject = row[0]
                        self.csv_content[subject] = {header[i]: row[i] for i in range(1, len(header))}

            if id_ == 'age':
                return np.asarray(float(self.csv_content[subject_id]['Age'])), None
            if id_ == 'resectionstatus':
                print("working...")
                status_class = ResectionStatusConversion.from_status(self.csv_content[subject_id]['ResectionStatus'])
                return np.asarray(status_class), None
            if id_ == 'survival':
                return np.asarray(int(self.csv_content[subject_id]['Survival'])), None

        raise ValueError('could not load {}'.format(file_path))



# class SurvivalDataLoader(miapy_fileload.Load):
#
#     def __init__(self):
#         self.csv_file = None
#         self.csv_content = None
#
#     def __call__(self, file_path: str, id_: str, category: str, subject_id: str
#                  ) -> typing.Tuple[np.ndarray, typing.Union[miapy_conv.ImageProperties, None]]:
#         if category == 'images':
#             img = sitk.ReadImage(file_path)
#             return sitk.GetArrayFromImage(img), miapy_conv.ImageProperties(img)
#
#         if category == 'labels':
#             img = sitk.ReadImage(file_path, sitk.sitkUInt8)
#             return sitk.GetArrayFromImage(img), miapy_conv.ImageProperties(img)
#
#         if category == 'age' or category == 'survival': # or category == 'survclass' or category == 'resectionstatus' :
#         #if category == 'age' or category == 'ResectionStatus' or category == 'Survival' or category == 'survclass' or category == 'volncr' or category == 'voled' or category == 'volet' or category == 'etrimwidth' or category == 'etgeomhet'                or category == 'rim_q1_clipped' or category == 'rim_q2_clipped' or category == 'rim_q3_clipped':
#                 #or category == 'logs1_glszm_LALGLE_T2_1' \
#                 #or category == 'logs1_gldm_SDHGLE_T2_1' or category == 'logs1_glszm_LALGLE_T2_4' or category == 'logs1_glszm_LALGLE_T2_2'\
#                 #or category == 'logs1_FOR_T2_2' or category == 'logs1_gldm_SDHGLE_T2_4' or category == 'logs1_gldm_SDHGLE_T2_2'\
#                 #or category == 'logs1_FOR_T2_4' or category == 'logs1_FOR_T2_1' or category == 'orig_glcm_SE_T1_2'\
#                 # or category == 'orig_glcm_SE_T1_1' or category == 'orig_glcm_SE_T1_4'
#
#             if self.csv_file is None or file_path != self.csv_file:
#                 self.csv_file = file_path
#                 self.csv_content = {}
#
#                 with open(self.csv_file) as f:
#                     reader = csv.reader(f)
#                     header = next(reader)
#                     for row in reader:
#                         subject = row[0]
#                         self.csv_content[subject] = {header[i]: row[i] for i in range(1, len(header))}
#             print(id_)
#             if id_ == 'age':
#                 return np.asarray(float(self.csv_content[subject_id]['Age'])), None
#             # if id_ == 'resectionstatus':
#             #     status_class = ResectionStatusConversion.from_status(
#             #         self.csv_content[subject_id]['ResectionStatus'])
#             #     return np.asarray(status_class), None
#             if id_ == 'survival':
#                 print("working...")
#                 return np.asarray(int(self.csv_content[subject_id]['Survival'])), None
#             #
#             #
#             #
#             # if id_ == 'survclass':
#             #     return np.asarray(int(self.csv_content[subject_id]['survclass'])), None
#             # if id_ == 'logs1_glszm_LALGLE_T2_1':
#             #     print("Still working...")
#             #     return np.asarray(float(self.csv_content[subject_id]['log-sigma-1-0-mm-3D_glszm_LargeAreaLowGrayLevelEmphasis_T2_l1nectrotic'])), None
#             # if id_ == 'logs1_gldm_SDHGLE_T2_1':
#             #     return np.asarray(float(self.csv_content[subject_id]['log-sigma-1-0-mm-3D_gldm_SmallDependenceHighGrayLevelEmphasis_T2_l1nectrotic'])), None
#             # if id_ == 'logs1_glszm_LALGLE_T2_4':
#             #     return np.asarray(float(self.csv_content[subject_id][
#             #                                   'log-sigma-1-0-mm-3D_glszm_LargeAreaLowGrayLevelEmphasis_T2_l4ce'])), None
#             # if id_ == 'logs1_glszm_LALGLE_T2_2':
#             #     return np.asarray(float(self.csv_content[subject_id][
#             #                                   'log-sigma-1-0-mm-3D_glszm_LargeAreaLowGrayLevelEmphasis_T2_l2edema'])), None
#             # if id_ == 'logs1_FOR_T2_2':
#             #     return np.asarray(float(self.csv_content[subject_id][
#             #                                   'log-sigma-1-0-mm-3D_firstorder_Range_T2_l2edema'])), None
#             #if id_ == 'logs1_gldm_SDHGLE_T2_4':
#             #    return np.asarray(int(self.csv_content[subject_id]['log-sigma-1-0-mm-3D_gldm_SmallDependenceHighGrayLevelEmphasis_T2_l4ce'])), None
#             # if id_ == 'logs1_gldm_SDHGLE_T2_2':
#             #     return np.asarray(float(self.csv_content[subject_id][
#             #                                   'log-sigma-1-0-mm-3D_gldm_SmallDependenceHighGrayLevelEmphasis_T2_l2edema'])), None
#             # if id_ == 'logs1_FOR_T2_4':
#             #     return np.asarray(float(self.csv_content[subject_id][
#             #                                   'log-sigma-1-0-mm-3D_firstorder_Range_T2_l4ce'])), None
#             # if id_ == 'logs1_FOR_T2_1':
#             #     return np.asarray(float(self.csv_content[subject_id][
#             #                                   'log-sigma-1-0-mm-3D_firstorder_Range_T2_l1nectrotic'])), None
#             # if id_ == 'orig_glcm_SE_T1_2':
#             #     return np.asarray(float(self.csv_content[subject_id][
#             #                                   'original_glcm_SumEntropy_T1_l2edema'])), None
#             # if id_ == 'orig_glcm_SE_T1_1':
#             #     return np.asarray(float(self.csv_content[subject_id][
#             #                                   'original_glcm_SumEntropy_T1_l1nectrotic'])), None
#             # if id_ == 'orig_glcm_SE_T1_4':
#             #     return np.asarray(float(self.csv_content[subject_id][
#             #                                   'original_glcm_SumEntropy_T1_l4ce'])), None
#             # if id_ == 'volncr':
#             #     return np.asarray(int(self.csv_content[subject_id]['volncr'])), None
#             # if id_ == 'voled':
#             #     return np.asarray(int(self.csv_content[subject_id]['voled'])), None
#             # if id_ == 'volet':
#             #     return np.asarray(int(self.csv_content[subject_id]['volet'])), None
#             # if id_ == 'etrimwidth':
#             #     return np.asarray(float(self.csv_content[subject_id]['etrimwidth'])), None
#             # if id_ == 'etgeomhet':
#             #     return np.asarray(float(self.csv_content[subject_id]['etgeomhet'])), None
#             # if id_ == 'rim_q1_clipped':
#             #     return np.asarray(float(self.csv_content[subject_id]['rim_q1_clipped'])), None
#             # if id_ == 'rim_q2_clipped':
#             #     return np.asarray(float(self.csv_content[subject_id]['rim_q2_clipped'])), None
#             # if id_ == 'rim_q3_clipped':
#             #     return np.asarray(float(self.csv_content[subject_id]['rim_q3_clipped'])), None
#
#         raise ValueError('could not load {}'.format(file_path))


class ResectionStatusConversion:

    @staticmethod
    def from_status(status: str) -> int:
        if status == 'NA':
            return 0
        elif status == 'GTR':
            return 1
        elif status == 'STR':
            return 2
        raise ValueError('unknown resection status "{}"'.format(status))

    @staticmethod
    def to_status(int_val: int) -> str:
        if int_val == 0:
            return 'NA'
        elif int_val == 1:
            return 'GTR'
        elif int_val == 2:
            return 'STR'
        raise ValueError('unknown resection class value "{}"'.format(int_val))


class DataStore:
    def __init__(self, hdf_file: str, data_transform: miapy_tfm.Transform = None):
        self.hdf_file = hdf_file
        self._dataset = None
        self._data_transform = data_transform

    def __del__(self):
        if self._dataset is not None:
            self._dataset.close_reader()

    def import_data(self, subjects: typing.List[miapy_data.SubjectFile], intensity_max: int, input_transform: miapy_tfm=None):
        with miapy_crt.get_writer(self.hdf_file) as writer:
            callbacks = miapy_crt.get_default_callbacks(writer)
            traverser = miapy_crt.SubjectFileTraverser()
            traverser.traverse(subjects, callback=callbacks, load=SurvivalDataLoader(), transform=input_transform)

    @property
    def dataset(self) -> miapy_extr.ParameterizableDataset:
        if self._dataset is None:
            self._dataset = miapy_extr.ParameterizableDataset(
                self.hdf_file,
                None,
                miapy_extr.SubjectExtractor(),
                self._data_transform)

        return self._dataset

    def set_transforms_enabled(self, enabled: bool):
        if enabled:
            self._dataset.set_transform(self._data_transform)
        else:
            self._dataset.set_transform(None)

    @staticmethod
    def collate_batch(batch) -> dict:
        # batch is a list of dicts -> change to dict of lists
        return dict(zip(batch[0], zip(*[d.values() for d in batch])))

    def get_all_metrics(self) -> typing.Tuple[np.ndarray, list]:
        """
        Get the metrics from all subjects and the corresponding column names
        
        :return: (metrics, column_names)
        """

        metrics = [self.dataset.direct_extract(
            miapy_extr.SelectiveDataExtractor(category=STORE_MORPHOMETRICS), idx) for idx in range(len(self.dataset))]
        column_names = self.dataset.reader.read(STORE_META_MORPHOMETRY_COLUMNS)

        return np.stack(self.collate_batch(metrics)[STORE_MORPHOMETRICS], axis=0), column_names

    # def get_intensity_scale_max(self) -> int:
    #     return self.dataset.reader.read(STORE_META_INTENSITY_RESCALE_MAX)

    def get_loader(self, batch_size: int, subject_ids: typing.List[str], num_workers: int):
        sampler = miapy_extr.SubsetRandomSampler(
            miapy_extr.select_indices(self.dataset, miapy_extr.SubjectSelection(subject_ids)))

        return miapy_extr.DataLoader(self.dataset,
                                     batch_size,
                                     sampler=sampler,
                                     collate_fn=self.collate_batch,
                                     num_workers=num_workers)


# def parse_subject_name(name: str) -> (str, str, str):
#     study = 'n/a'
#     age = 'n/a'
#     sex = 'n/a'
#     study_idx = 1
#     sex_idx = 2
#     age_idx = 3
#     # e.g. MS_HC036_f_26_MPRfischl
#     matcher = re.match('([^_]+)_[^_]+_([mfw])_([0-9]+)_.*', name)
#     if matcher is None:
#         # try age and sex swapped
#         matcher = re.match('([^_]+)_[^_]+_([0-9]+)_([mfw])_.*', name)
#         sex_idx = 3
#         age_idx = 2
#     if matcher is None:
#         # try without study prefix (e.g. P203pat103_w_15_MDEFT)
#         matcher = re.match('[^_]+_([mfw])_([0-9]+)_.*', name)
#         study_idx = -1
#         sex_idx = 1
#         age_idx = 2
#     if matcher is None:
#         if not name.startswith('ADNI'):
#             print('WARN: Unknown age/sex for {}'.format(name))
#         study = re.match('([^_]+)_.*', name).group(1)
#     else:
#         if study_idx > 0:
#             study = matcher.group(study_idx)
#         sex = matcher.group(sex_idx).replace('w', 'f')
#         age = matcher.group(age_idx)
#
#     return study, age, sex
