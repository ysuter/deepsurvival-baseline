import os
import shutil
import csv
import glob

in_dirs = ('/home/j/Desktop/brats_dc/Training/LGG', '/home/j/Desktop/brats_dc/Training/HGG')
in_csv = '/home/j/Desktop/brats_dc/Training/survival_data.csv'
out_dir = '/media/data/datasets/Brats18_survival'

if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

# collect all subject directories
all_subjects = {}
for in_dir in in_dirs:
    subject_dirs = glob.glob(in_dir + '/*')
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        all_subjects[subject_id] = subject_dir

# select the survival subjects
survival_subjects = {}
with open(in_csv, newline='') as f:
    reader = csv.reader(f)
    next(reader)  # ignore header
    for row in reader:
        subject_id = row[0]
        if subject_id not in all_subjects:
            raise ValueError('subject {} not found in dataset'.format(subject_id))
        survival_subjects[subject_id] = all_subjects[subject_id]


# copy files to output directory
for in_subject_dir in survival_subjects.values():
    out_subject_dir = os.path.join(out_dir, os.path.basename(in_subject_dir))
    shutil.copytree(in_subject_dir, out_subject_dir)

out_csv = os.path.join(out_dir, os.path.basename(in_csv))
shutil.copy(in_csv, out_csv)


