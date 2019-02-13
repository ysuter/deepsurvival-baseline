import csv
import pandas as pd
import argparse

def main(inpcsv1: str, inpcsv2: str, matchcolname: str, dropcols1: list, dropcols2: list, outcsv: str, mode: str):

    csv1 = pd.read_csv(inpcsv1)
    csv2 = pd.read_csv(inpcsv2)

    # Left merge: Keep everything from the left, drop things from the right (if they don't match)
    # drop cols in csv1
    if len(dropcols1) != 0:
        csv1 = csv1.drop(dropcols1, axis=1)
    if len(dropcols2) != 0:
        csv2 = csv2.drop(dropcols2, axis=1)
    df_merged = pd.merge(left=csv1, right=csv2, on=matchcolname, how=mode, indicator=False)

    df_merged.to_csv(outcsv, sep=',', index=False)

    # sg = []
    # fqdn = {}
    # output = []
    # with open(inpcsv1, 'rt') as src:
    #     read = csv.reader(src, delimiter=',')
    #     for dataset in read:
    #         sg.append(dataset)
    #
    # with open(inpcsv2, 'rt') as src1:
    #     read1 = csv.reader(src1, delimiter=',')
    #     print(read1)
    #     for to_append, to_match in read1:
    #         fqdn[to_match] = to_append
    #
    # for dataset in sg:
    #     to_append = fqdn.get(dataset[2])  # If the key matched, to_append now contains the string to append, else it becomes None
    #     if to_append:
    #         dataset.append(to_append)  # Append the field
    #         output.append(dataset)  # Append the row to the result list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge two csv files with row matching.')

    parser.add_argument(
        '--inpcsv1',
        type=str,
        help='Path to first csv file'
    )

    parser.add_argument(
        '--inpcsv2',
        type=str,
        help='Path to second csv file'
    )

    parser.add_argument(
        '--matchcolname',
        type=str,
        default='SubjID',
        help='Column names to match in csv files'
    )

    parser.add_argument(
        '--dropcols1',
        default=[],
        type=list,
        help='List of columns to drop in csv1'
    )

    parser.add_argument(
        '--dropcols2',
        default=[],
        type=list,
        help='List of columns to drop in csv2'
    )

    parser.add_argument(
        '--outcsv',
        type=str,
        help='Path for output csv file'
    )

    parser.add_argument(
        '--mode',
        default='left',
        type=str,
        help='Specifiy of left or right merge is performed'
    )

    inp1 = '/run/user/1000/gvfs/smb-share:server=istb-brain,share=data/mia/BrainOncology/BraTS/Brats2018/Features/pyradiomics_corrected/pyrad_train_int_features_top40.csv'

    inp2 = '/run/user/1000/gvfs/smb-share:server=istb-brain,share=data/mia/BrainOncology/BraTS/Brats2018/Features/shapeOWN_train_volumeratios.csv'

    out = '/home/yannick/Dropbox/PyRad_Featureextractor/featureselection/Subset/subset1.csv'


    args = parser.parse_args()
    # main(args.inpcsv1, args.inpcsv2, args.matchcol1, args.matchcol2, args.dropcols1, args.dropcols2, args.outcsv)
    main(inp1, inp2, 'SubjID', [], [], out, 'left')


