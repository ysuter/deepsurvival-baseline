import csv
import typing
import io
import numpy as np
from sklearn import metrics


def parse_results_csv(csv_file) -> typing.Tuple[list, list, np.ndarray, np.ndarray]:
    params = None
    subjects = []
    predictions = []
    gt = []
    # name, study, age, sex, col1, col1.gt, col2, col2.gt....
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if params is None:
                params = row[4::2]
            else:
                subjects.append(row[0])
                predictions.append(row[4::2])
                gt.append(row[5::2])

    predictions = np.array(predictions).astype(np.float32)
    gt = np.array(gt).astype(np.float32)

    return subjects, params, predictions, gt


def print_summary(subjects: list, params: list, predictions: np.ndarray, gt: np.ndarray, all_subjects=False) -> str:
    buff = io.StringIO()
    # scores per subject
    r2_scores_subject = []
    rmse_scores_subject = []
    for idx, _ in enumerate(subjects):
        r2_scores_subject.append(metrics.r2_score(gt[idx], predictions[idx]))
        rmse_scores_subject.append(np.sqrt(metrics.mean_squared_error(gt[idx], predictions[idx])))

    r2_scores_subject = np.array(r2_scores_subject)
    rmse_scores_subject = np.array(rmse_scores_subject)

    # scores per metric
    r2_scores_metric = []
    rmse_scores_metric = []
    for idx, _ in enumerate(params):
        r2_scores_metric.append(metrics.r2_score(gt[:, idx], predictions[:, idx]))
        rmse_scores_metric.append(np.sqrt(metrics.mean_squared_error(gt[:, idx], predictions[:, idx])))

    r2_scores_metric = np.array(r2_scores_metric)
    rmse_scores_metric = np.array(rmse_scores_metric)

    # print scores per metric
    print('{:<40}{:<10}  {}'.format('Metric', 'R^2', 'RMSE'), file=buff)
    print('------------------------------------------------------', file=buff)
    for idx, _ in enumerate(params):
        print('{:<40}{:.8f}  {:.3f}'.format(params[idx], r2_scores_metric[idx], rmse_scores_metric[idx]), file=buff)

    flop_metrics = np.argsort(r2_scores_metric)
    print(file=buff)

    # print top 5 and flop 5 subjects
    flop_subjects = np.argsort(r2_scores_subject)
    n_flop_subjects_5 = min(5, max(1, int(np.floor(len(flop_subjects) / 3))))
    if all_subjects:
        n_flop_subjects = len(flop_subjects)
    else:
        n_flop_subjects = n_flop_subjects_5

    print('{:<40}{:<10}  {}'.format('Top 5 (by R^2)', 'R^2', 'RMSE'), file=buff)
    print('--------------------------------------------------------------------------', file=buff)
    for idx in range(len(flop_subjects)-1, len(flop_subjects)-n_flop_subjects-1, -1):
        subj_idx = flop_subjects[idx]
        print('{:<40}{:.8f}  {:.3f}'.format(
            subjects[subj_idx], r2_scores_subject[subj_idx], rmse_scores_subject[subj_idx]), file=buff)

    if not all_subjects:
        print(file=buff)
        print('{:<40}{:<10}  {}'.format('Flop 5 (by R^2)', 'R^2', 'RMSE'), file=buff)
        print('--------------------------------------------------------------------------', file=buff)
        for idx in range(0, n_flop_subjects):
            subj_idx = flop_subjects[idx]
            print('{:<40}{:.8f}  {:.3f}'.format(
                subjects[subj_idx], r2_scores_subject[subj_idx], rmse_scores_subject[subj_idx]), file=buff)

    print(file=buff)
    print(file=buff)
    print('R^2 Total:                {:.8f}'.format(metrics.r2_score(gt, predictions)), file=buff)
    print('R^2 excl. flop5 subjects: {:.8f}'.format(metrics.r2_score(gt[flop_subjects[n_flop_subjects_5:]],
                                                                     predictions[flop_subjects[n_flop_subjects_5:]])),file=buff)
    if len(params) > 1:
        n_flop_metric = min(5, max(1, int(np.floor(len(params) / 3))))
        print('R^2 excl. flop{} metrics:  {:.8f}'.format(n_flop_metric,
                                                         metrics.r2_score(gt[:, flop_metrics[n_flop_metric:]],
                                                                          predictions[:, flop_metrics[n_flop_metric:]])), file=buff)

    # for cortex also report cortical Thickness and Curvature separately
    for p in ['ThickAvg', 'MeanCurv']:
        # get indices of params containing p
        indices = [idx for idx, elem in enumerate(params) if p in elem]
        if len(indices) > 0:
            gt_sub = gt[:, indices]
            predictions_sub = predictions[:, indices]
            print('R^2 {}:             {:.8f}'.format(p, metrics.r2_score(gt_sub, predictions_sub)), file=buff)

    return buff.getvalue()
