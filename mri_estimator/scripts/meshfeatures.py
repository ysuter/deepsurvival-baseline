
import miapy.plotting.surfacedistance as miapy_surfdist
import os
import SimpleITK as sitk
import numpy as np
import csv
import math
import vtk
from vtk.util import numpy_support
import matplotlib.pyplot as plt

# inputcsv = '/home/yannick/Dropbox/Doktorat/BraTS/Testeval/survival_evaluation.csv'
inputcsv = '/run/user/1000/gvfs/smb-share:server=istb-brain,share=data/mia/BrainOncology/BraTS/Brats2018/Validation/Brats18Validation_onlySurvival/survival_evaluation.csv'
# inputcsv = '/run/user/1000/gvfs/smb-share:server=istb-brain,share=data/mia/BrainOncology/BraTS/Brats2018/Testing/survival_evaluation.csv'

csvoutput = '/home/yannick/Dropbox/Doktorat/BraTS/Testeval/shapeOWN_valnew.csv'
#csvoutput_df = '/home/yannick/Dropbox/Doktorat/BraTS/shape_df.csv'
#dataroot = '/home/yannick/Documents/WMShared/BraTS2018_Training/Training/HGG'
#bratumialabelroot = '/home/yannick/Documents/WMShared/BraTS2018_Training/Training/Labels_bratumia'
nifitylabelroot = '/run/user/1000/gvfs/smb-share:server=istb-brain,share=data/mia/BrainOncology/BraTS/Brats2018/Validation/niftynet_segmentations'
# nifitylabelroot = '/run/user/1000/gvfs/smb-share:server=istb-brain,share=data/mia/BrainOncology/BraTS/Brats2018/Testing_seg/niftynet_segmentations'

# read from input csv
with open(inputcsv, 'r') as readFile:
    reader = csv.reader(readFile)
    lines = list(reader)
    # lines[2] = row

headers = lines[0]

newheaders = lines[0] + ['gt_volncr', 'gt_voled', 'gt_volet', 'gt_etrimwidth', 'gt_etgeomhet', 'gt_rim_q1', 'gt_rim_q2', 'gt_rim_q3', 'gt_etgeomhet_clipped', 'gt_rim_q1_clipped', 'gt_rim_q2_clipped', 'gt_rim_q3_clipped',
              'bratumia_volncr', 'bratumia_voled', 'bratumia_volet', 'bratumia_etrimwidth', 'bratumia_etgeomhet', 'bratumia_rim_q1', 'bratumia_rim_q2', 'bratumia_rim_q3', 'bratumia_etgeomhet_clipped', 'bratumia_rim_q1_clipped', 'bratumia_rim_q2_clipped', 'bratumia_rim_q3_clipped',
              'niftynet_volncr', 'niftynet_voled', 'niftynet_volet', 'niftynet_etrimwidth', 'niftynet_etgeomhet', 'niftynet_rim_q1', 'niftynet_rim_q2', 'niftynet_rim_q3', 'niftynet_etgeomhet_clipped', 'niftynet_rim_q1_clipped', 'niftynet_rim_q2_clipped', 'niftynet_rim_q3_clipped']
# newheaders = lines[0] + ['gt_volncr', 'gt_voled', 'gt_volet', 'gt_etrimwidth', 'gt_etgeomhet', 'gt_rim_q1', 'gt_rim_q2', 'gt_rim_q3', 'gt_etgeomhet_clipped', 'gt_rim_q1_clipped', 'gt_rim_q2_clipped', 'gt_rim_q3_clipped',
#               'bratumia_volncr', 'bratumia_voled', 'bratumia_volet', 'bratumia_etrimwidth', 'bratumia_etgeomhet', 'bratumia_rim_q1', 'bratumia_rim_q2', 'bratumia_rim_q3', 'bratumia_etgeomhet_clipped', 'bratumia_rim_q1_clipped', 'bratumia_rim_q2_clipped', 'bratumia_rim_q3_clipped',
#               'niftynet_volncr', 'niftynet_voled', 'niftynet_volet', 'niftynet_etrimwidth', 'niftynet_etgeomhet', 'niftynet_rim_q1', 'niftynet_rim_q2', 'niftynet_rim_q3', 'niftynet_etgeomhet_clipped', 'niftynet_rim_q1_clipped', 'niftynet_rim_q2_clipped', 'niftynet_rim_q3_clipped']

newheaders = lines[0] + ['volncr', 'voled', 'volet', 'etrimwidth', 'etgeomhet', 'rim_q1', 'rim_q2', 'rim_q3', 'etgeomhet_clipped', 'rim_q1_clipped', 'rim_q2_clipped', 'rim_q3_clipped']
#newheaders_df = ['seg'] + lines[0] + ['volncr', 'voled', 'volet', 'etrimwidth', 'etgeomhet', 'rim_q1', 'rim_q2', 'rim_q3', 'etgeomhet_clipped', 'rim_q1_clipped', 'rim_q2_clipped', 'rim_q3_clipped']


print(newheaders)

#segtypes = ['GT', 'Bratumia', 'Niftynet']
segtypes = ['Niftynet']

with open(csvoutput, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(newheaders)

#with open(csvoutput_df, 'w') as writeFile2:
    #writer2 = csv.writer(writeFile2)
    #writer2.writerow(newheaders_df)

# loop through every line and load the 1) GT segmentation and 2) BraTuMIA segmentation
for lineidx, currentline in enumerate(lines[1:]):
    currentsubj = currentline[0]
    print(currentline)

    # load GT label map
    #gtlabelpath = os.path.join(dataroot, currentsubj, currentsubj + '_seg.nii.gz')
    #bratumialabelpath = os.path.join(bratumialabelroot, currentsubj + '_seg_bratumia.nii.gz')
    niftylabelpath = os.path.join(nifitylabelroot, currentsubj + '.nii.gz')

    #gtlabel = sitk.ReadImage(gtlabelpath)
    #bratumialabel = sitk.ReadImage(bratumialabelpath)
    niftylabel = sitk.ReadImage(niftylabelpath)
    #
    # # extract information for the ground truth segmentation
    # gtvoxelsize = gtlabel.GetSpacing()
    # voxelvol = gtvoxelsize[0] * gtvoxelsize[1] * gtvoxelsize[2]
    # gtlabelarr = sitk.GetArrayFromImage(gtlabel)
    # gtnecroticvol = np.count_nonzero(gtlabelarr == 1) * voxelvol
    # gtedemavol = np.count_nonzero(gtlabelarr == 2) * voxelvol
    # gtenhvol = np.count_nonzero(gtlabelarr == 4) * voxelvol
    # gtenhrimwidth = (3./(4*math.pi)) ** (1. / 3) * ((gtenhvol + gtnecroticvol) ** (1./3) - (gtnecroticvol) ** (1./3))
    #
    # # extract information for the bratumia segmentation
    # b_voxelsize = bratumialabel.GetSpacing()
    # b_voxelvol = b_voxelsize[0] * b_voxelsize[1] * b_voxelsize[2]
    # b_labelarr = sitk.GetArrayFromImage(bratumialabel)
    # b_necroticvol = np.count_nonzero(b_labelarr == 1) * b_voxelvol
    # b_edemavol = np.count_nonzero(b_labelarr == 2) * b_voxelvol
    # b_enhvol = np.count_nonzero(b_labelarr == 4) * b_voxelvol
    # b_enhrimwidth = (3./(4*math.pi)) ** (1. / 3) * ((b_enhvol + b_necroticvol) ** (1./3) - (b_necroticvol) ** (1./3))

    # extract information for the bratumia segmentation
    n_voxelsize = niftylabel.GetSpacing()
    n_voxelvol = n_voxelsize[0] * n_voxelsize[1] * n_voxelsize[2]
    n_labelarr = sitk.GetArrayFromImage(niftylabel)
    n_necroticvol = np.count_nonzero(n_labelarr == 1) * n_voxelvol
    n_edemavol = np.count_nonzero(n_labelarr == 2) * n_voxelvol
    n_enhvol = np.count_nonzero(n_labelarr == 4) * n_voxelvol
    n_enhrimwidth = (3./(4.*math.pi)) ** (1. / 3) * ((n_enhvol + n_necroticvol) ** (1./3) - (n_necroticvol) ** (1./3))

    ## calculate geometric heterogeneity
    # read label map as vtk files
    # gt_vtklabel, gt_tfm = miapy_surfdist.read_and_transform_image(gtlabelpath)
    # b_vtklabel, b_tfm = miapy_surfdist.read_and_transform_image(bratumialabelpath)
    n_vtklabel, n_tfm = miapy_surfdist.read_and_transform_image(niftylabelpath)

    # tmp_gtvtkreader = vtk.vtkNIFTIImageReader()
    # tmp_gtvtkreader.SetFileName(gtlabelpath)
    # tmp_gtvtkreader.Update()
    #
    # tmp_bvtkreader = vtk.vtkNIFTIImageReader()
    # tmp_bvtkreader.SetFileName(bratumialabelpath)
    # tmp_bvtkreader.Update()

    tmp_nvtkreader = vtk.vtkNIFTIImageReader()
    tmp_nvtkreader.SetFileName(niftylabelpath)
    tmp_nvtkreader.Update()

    # # threshold image to combine NCR/NET and CET labels to later get an "outer" surface isocontour
    # thresholder = vtk.vtkImageThreshold()
    # thresholder.ReplaceInOn()
    # thresholder.SetInputData(gt_vtklabel)
    # thresholder.ThresholdBetween(1, 1) # Label for contrast enhancing tumor
    # thresholder.SetInValue(4)          # Set to 4 (same as CET label)
    # thresholder.SetOutputScalarTypeToUnsignedShort()
    # thresholder.Update()
    #
    #
    # # threshold image to combine NCR/NET and CET labels to later get an "outer" surface isocontour
    # thresholder_b = vtk.vtkImageThreshold()
    # thresholder_b.ReplaceInOn()
    # thresholder_b.SetInputData(b_vtklabel)
    # thresholder_b.ThresholdBetween(1, 1) # Label for contrast enhancing tumor
    # thresholder_b.SetInValue(4)          # Set to 4 (same as CET label)
    # thresholder_b.SetOutputScalarTypeToUnsignedShort()
    # thresholder_b.Update()

    # threshold image to combine NCR/NET and CET labels to later get an "outer" surface isocontour
    thresholder_n = vtk.vtkImageThreshold()
    thresholder_n.ReplaceInOn()
    thresholder_n.SetInputData(n_vtklabel)
    thresholder_n.ThresholdBetween(1, 1) # Label for contrast enhancing tumor
    thresholder_n.SetInValue(4)          # Set to 4 (same as CET label)
    thresholder_n.SetOutputScalarTypeToUnsignedShort()
    thresholder_n.Update()


    # b_vtkthresholded = thresholder_b.GetOutput()
    # save combined label for inspection
    # writer = vtk.vtkNIFTIImageWriter()
    # writer.SetNIFTIHeader(tmp_bvtkreader.GetNIFTIHeader())
    # writer.SetInputConnection(thresholder_b_netncr.GetOutputPort())
    # writer.SetFileName(os.path.join(dataroot, currentsubj, currentsubj + '_bratumiaseg_netncrcombined.nii.gz'))
    # writer.Update()
    #
    # writer.SetInputConnection(thresholder.GetOutputPort())
    # writer.SetFileName(os.path.join(dataroot, currentsubj, currentsubj + '_bratumiaseg_netncrcetcombined.nii.gz'))
    # writer.Update()

    #
    # gt_CETiso = miapy_surfdist.compute_isocontour(thresholder.GetOutput(), 4)
    # gt_NCRiso = miapy_surfdist.compute_isocontour(gt_vtklabel, 1)
    #
    # b_CETiso = miapy_surfdist.compute_isocontour(thresholder_b.GetOutput(), 4)
    # b_NCRiso = miapy_surfdist.compute_isocontour(b_vtklabel, 1)

    n_CETiso = miapy_surfdist.compute_isocontour(thresholder_n.GetOutput(), 4)
    n_NCRiso = miapy_surfdist.compute_isocontour(n_vtklabel, 1)

    # Transform polydata to match physical space according to NifTi file
    # gt_tfm_inv = gt_tfm.Inverse()

    # tfmfilter = vtk.vtkTransformPolyDataFilter()
    # tfmfilter.SetTransform(gt_tfm)
    # tfmfilter.SetInputData(gt_CETiso)
    # tfmfilter.Update()
    # gt_CETiso_tfm =  tfmfilter.GetOutput()
    #
    # tfmfilter.SetInputData(gt_NCRiso)
    # tfmfilter.Update()
    # gt_NCRiso_tfm =  tfmfilter.GetOutput()


    # # save polydata for inspection
    # vtkwriter = vtk.vtkPolyDataWriter()
    # vtkwriter.SetInputData(gt_CETiso)
    # vtkwriter.SetFileName(os.path.join(dataroot, currentsubj, currentsubj + '_gt_CETiso.vtk'))
    # vtkwriter.Write()
    # vtkwriter.SetInputData(gt_NCRiso)
    # vtkwriter.SetFileName(os.path.join(dataroot, currentsubj, currentsubj + '_gt_NCRiso.vtk'))
    # vtkwriter.Write()

    # surfdist_gt = miapy_surfdist.compute_surface_to_surface_distance(gt_NCRiso, gt_CETiso)
    # surfdist_b = miapy_surfdist.compute_surface_to_surface_distance(b_NCRiso, b_CETiso)
    surfdist_n = miapy_surfdist.compute_surface_to_surface_distance(n_NCRiso, n_CETiso)
    #
    # gt_pt = surfdist_gt.GetPointData()
    # gtdistscalars = gt_pt.GetScalars()
    # gtdistarray = numpy_support.vtk_to_numpy(gtdistscalars)
    # gtdistarray_clipped = gtdistarray[gtdistarray > 0]
    #
    # gh_gt_clipped = (np.percentile(gtdistarray_clipped,100) - np.percentile(gtdistarray_clipped,75))/np.percentile(gtdistarray_clipped,100)
    # gt_rim_qartile1_clipped = np.percentile(gtdistarray_clipped, 25)
    # gt_rim_qartile2_clipped = np.percentile(gtdistarray_clipped, 50)
    # gt_rim_qartile3_clipped = np.percentile(gtdistarray_clipped, 75)
    #
    # gh_gt = (np.percentile(gtdistarray,100) - np.percentile(gtdistarray,75))/np.percentile(gtdistarray,100)
    # gt_rim_qartile1 = np.percentile(gtdistarray, 25)
    # gt_rim_qartile2 = np.percentile(gtdistarray, 50)
    # gt_rim_qartile3 = np.percentile(gtdistarray, 75)
    # #
    # # print(gtdistarray)
    # # print("Mean: " +str(np.mean(gtdistarray)))
    # # print("Median: " +str(np.median(gtdistarray)))
    # # print("Min: " +str(np.min(gtdistarray)))
    # # print("Max: " +str(np.max(gtdistarray)))
    #
    # b_pt = surfdist_b.GetPointData()
    # b_distscalars = b_pt.GetScalars()
    # b_distarray = numpy_support.vtk_to_numpy(b_distscalars)
    # b_distarray_clipped = b_distarray[b_distarray > 0]
    #
    # gh_b_clipped = (np.percentile(b_distarray_clipped,100) - np.percentile(b_distarray_clipped,75))/np.percentile(b_distarray_clipped,100)
    # b_rim_qartile1_clipped = np.percentile(b_distarray_clipped, 25)
    # b_rim_qartile2_clipped = np.percentile(b_distarray_clipped, 50)
    # b_rim_qartile3_clipped = np.percentile(b_distarray_clipped, 75)
    #
    # gh_b = (np.percentile(b_distarray,100) - np.percentile(b_distarray,75))/np.percentile(b_distarray,100)
    # b_rim_qartile1 = np.percentile(b_distarray, 25)
    # b_rim_qartile2 = np.percentile(b_distarray, 50)
    # b_rim_qartile3 = np.percentile(b_distarray, 75)
    #
    # # print(b_distarray)
    # # print("Mean: " +str(np.mean(b_distarray)))
    # # print("Median: " +str(np.median(b_distarray)))
    # # print("Min: " +str(np.min(b_distarray)))
    # # print("Max: " +str(np.max(b_distarray)))
    # # n, bins, patches = plt.hist(b_distarray) #, 50, normed=1, facecolor='green', alpha=0.75)
    # # plt.show()

    n_pt = surfdist_n.GetPointData()
    n_distscalars = n_pt.GetScalars()
    n_distarray = numpy_support.vtk_to_numpy(n_distscalars)
    n_distarray_clipped = n_distarray[n_distarray > 0]

    gh_n_clipped = (np.percentile(n_distarray_clipped,100) - np.percentile(n_distarray_clipped,75))/np.percentile(n_distarray_clipped,100)
    n_rim_qartile1_clipped = np.percentile(n_distarray_clipped, 25)
    n_rim_qartile2_clipped = np.percentile(n_distarray_clipped, 50)
    n_rim_qartile3_clipped = np.percentile(n_distarray_clipped, 75)

    gh_n = (np.percentile(n_distarray,100) - np.percentile(n_distarray,75))/np.percentile(n_distarray,100)
    n_rim_qartile1 = np.percentile(n_distarray, 25)
    n_rim_qartile2 = np.percentile(n_distarray, 50)
    n_rim_qartile3 = np.percentile(n_distarray, 75)

    csvline = currentline + [n_necroticvol, n_edemavol, n_enhvol, n_enhrimwidth, gh_n, n_rim_qartile1, n_rim_qartile2, n_rim_qartile3, gh_n_clipped, n_rim_qartile1_clipped, n_rim_qartile2_clipped, n_rim_qartile3_clipped]
   # csvline = currentline + [gtnecroticvol, gtedemavol, gtenhvol, gtenhrimwidth, gh_gt, gt_rim_qartile1, gt_rim_qartile2, gt_rim_qartile3, gh_gt_clipped, gt_rim_qartile1_clipped, gt_rim_qartile2_clipped, gt_rim_qartile3_clipped,
   #                           b_necroticvol, b_edemavol, b_enhvol, b_enhrimwidth, gh_b, b_rim_qartile1, b_rim_qartile2, b_rim_qartile3, gh_b_clipped, b_rim_qartile1_clipped, b_rim_qartile2_clipped, b_rim_qartile3_clipped,
   #                           n_necroticvol, n_edemavol, n_enhvol, n_enhrimwidth, gh_n, n_rim_qartile1, n_rim_qartile2, n_rim_qartile3, gh_n_clipped, n_rim_qartile1_clipped, n_rim_qartile2_clipped, n_rim_qartile3_clipped]

    # csvline_df = [[0 for x in range(len(newheaders_df))] for x in range(len(segtypes))]

    # csvline_df_gt = ['GT'] + currentline + [gtnecroticvol, gtedemavol, gtenhvol, gtenhrimwidth, gh_gt,
    #                                                      gt_rim_qartile1, gt_rim_qartile2, gt_rim_qartile3,
    #                                                      gh_gt_clipped, gt_rim_qartile1_clipped,
    #                                                      gt_rim_qartile2_clipped, gt_rim_qartile3_clipped]
    # csvline_df_b = ['Bratumia'] + currentline + [b_necroticvol, b_edemavol, b_enhvol, b_enhrimwidth, gh_b, b_rim_qartile1,
    #                               b_rim_qartile2, b_rim_qartile3, gh_b_clipped, b_rim_qartile1_clipped,
    #                               b_rim_qartile2_clipped, b_rim_qartile3_clipped]
    # csvline_df_n = ['Niftynet'] + currentline + [n_necroticvol, n_edemavol, n_enhvol, n_enhrimwidth, gh_n, n_rim_qartile1, n_rim_qartile2, n_rim_qartile3, gh_n_clipped, n_rim_qartile1_clipped, n_rim_qartile2_clipped, n_rim_qartile3_clipped]
    #
    # with open(csvoutput_df, 'a') as writeFile2:
    #     writer3 = csv.writer(writeFile2)
    #     writer3.writerow(csvline_df_gt)
    #     writer3.writerow(csvline_df_b)
    #     writer3.writerow(csvline_df_n)

    #csvline_df[1].append(['Bratumia'] + currentline + [b_necroticvol, b_edemavol, b_enhvol, b_enhrimwidth, gh_b, b_rim_qartile1,
    #                              b_rim_qartile2, b_rim_qartile3, gh_b_clipped, b_rim_qartile1_clipped,
    #                              b_rim_qartile2_clipped, b_rim_qartile3_clipped])
    #csvline_df[2].append(['Niftynet'] + currentline + [n_necroticvol, n_edemavol, n_enhvol, n_enhrimwidth, gh_b, n_rim_qartile1, n_rim_qartile2, n_rim_qartile3, gh_n_clipped, n_rim_qartile1_clipped, n_rim_qartile2_clipped, n_rim_qartile3_clipped])

    #csvline_df = csvline_df + ['Bratumia'] + currentline + [b_necroticvol, b_edemavol, b_enhvol, b_enhrimwidth, gh_b, b_rim_qartile1, b_rim_qartile2, b_rim_qartile3, gh_b_clipped, b_rim_qartile1_clipped, b_rim_qartile2_clipped, b_rim_qartile3_clipped]

    #csvline_df = csvline_df + ['Niftynet'] + currentline + [n_necroticvol, n_edemavol, n_enhvol, n_enhrimwidth, gh_b, n_rim_qartile1, n_rim_qartile2, n_rim_qartile3, gh_n_clipped, n_rim_qartile1_clipped, n_rim_qartile2_clipped, n_rim_qartile3_clipped]

    #csvline_df = csvline_df.reshape((3,len(newheaders_df)))
    #print(csvline_df)
    #csvline_df[0][:] = str(segtypes[0]) + currentline + [gtnecroticvol, gtedemavol, gtenhvol, gtenhrimwidth, gh_gt, gt_rim_qartile1, gt_rim_qartile2, gt_rim_qartile3, gh_gt_clipped, gt_rim_qartile1_clipped, gt_rim_qartile2_clipped, gt_rim_qartile3_clipped]
    #csvline_df[1][:] = str(segtypes[1]) + currentline + [b_necroticvol, b_edemavol, b_enhvol, b_enhrimwidth, gh_b, b_rim_qartile1, b_rim_qartile2, b_rim_qartile3, gh_b_clipped, b_rim_qartile1_clipped, b_rim_qartile2_clipped, b_rim_qartile3_clipped]
    #csvline_df[2][:] = str(segtypes[2]) + currentline + [n_necroticvol, n_edemavol, n_enhvol, n_enhrimwidth, gh_b, n_rim_qartile1, n_rim_qartile2, n_rim_qartile3, gh_n_clipped, n_rim_qartile1_clipped, n_rim_qartile2_clipped, n_rim_qartile3_clipped]


    with open(csvoutput, 'a') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(csvline)

    #with open(csvoutput_df, 'a') as writeFile2:
    #    writer3 = csv.writer(writeFile2)
    #    writer3.writerows(csvline_df)

    # vtkwriter.SetInputData(surfdist_gt)
    # vtkwriter.SetFileName(os.path.join(dataroot, currentsubj, currentsubj + '_gt_surfist_CET_NCR.vtk'))
    # vtkwriter.Write()

readFile.close()
writeFile.close()
