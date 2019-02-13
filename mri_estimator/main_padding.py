import SimpleITK as sitk
import argparse

def main(padding, in_file, out_file):
    img = sitk.ReadImage(in_file)

    cpif = sitk.ConstantPadImageFilter()
    cpif.SetPadLowerBound([padding, padding, padding])
    cpif.SetPadUpperBound([padding, padding, padding])
    img_new = cpif.Execute(img)

    sitk.WriteImage(img_new, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-Padding MRI in each direction by given voxels')

    parser.add_argument(
        'padding',
        type=int,
        help='Number of voxels to add.'
    )

    parser.add_argument(
        'in_file',
        type=str,
        help='Input volume'
    )

    parser.add_argument(
        'out_file',
        type=str,
        help='Output volume'
    )

    args = parser.parse_args()
    main(args.padding, args.in_file, args.out_file)
