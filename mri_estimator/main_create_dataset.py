import os
import argparse
import numpy as np

import data.collector as data_collector
import data.storage as data_storage
import data.preprocess as data_preproc
import miapy.data.transformation as miapy_tfm


def assert_exists(file: str):
    if not os.path.exists(file):
        print('Error: {} not found'.format(file))
        exit(1)


class DTypeTransform(miapy_tfm.Transform):

    def __init__(self, dtype, entries=('images',)) -> None:
        self.dtype = dtype
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                continue

            np_entry = sample[entry]
            sample[entry] = np_entry.astype(self.dtype)

        return sample


def main(hdf_file: str, intensity_rescale_max: int, normalize_zscore: bool, center: bool, data_dir: str):
    transformers = []

    if center:
        print('Applying center centroid')
        transformers.append(data_preproc.CenterCentroidTransform())

    if intensity_rescale_max:
        print('Applying intensity rescale {}'.format(intensity_rescale_max))
        transformers.append(miapy_tfm.IntensityRescale(0, intensity_rescale_max))
        if intensity_rescale_max <= 255:
            transformers.append(DTypeTransform(np.uint8))
        else:
            transformers.append(DTypeTransform(np.float16))
    if normalize_zscore:
        transformers.append(miapy_tfm.IntensityNormalization())

    transformer = miapy_tfm.ComposeTransform(transformers)

    collector = data_collector.Brats18Collector(data_dir)

    if os.path.exists(hdf_file):
        print('Overriding existing {}'.format(hdf_file))
        os.remove(hdf_file)

    store = data_storage.DataStore(hdf_file)
    store.import_data(collector.subject_files, intensity_rescale_max, normalize_zscore, transformer)

    print('{} subjects imported to {}'.format(len(collector.subject_files), hdf_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset (.h5) from one or more directories with subjects.')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='out/datasetwlabels_normit4095_v5.h5',
        help='Path to hd5 file.'
    )

    parser.add_argument(
        '--intensity_rescale_max',
        type=int,
        default=4095,
        help='Apply intensity rescale(0, <max>) to input data and convert to uint8 (<=255) or float16, e.g. 255 or 4095'
    )

    parser.add_argument(
        '--normalize_zscore',
        type=bool,
        default=False,
        help='Apply normalization to input data (x-mean(x))/(std(x))'
    )
    parser.add_argument(
        '--center',
        type=bool,
        default=True,
        help='Center the centroid.'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        metavar='dir',
        #
        # required=True,
        default='/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/Brats18/Data/TrainVal',
        help='Directories to import subjects from.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.intensity_rescale_max, args.normalize_zscore, args.center, args.data_dir)
