import os
import timeit
import importlib
import inspect
import datetime
import time
import csv
import re
import typing
import logging
import numpy as np
import tensorflow as tf
import torch
import random
import scipy
from sklearn import model_selection
from sklearn import metrics
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
#import sys
#sys.path.append('/hd/BRATS/Brats18/mri_estimator/scripts')
#from jointplot_w_hue import jointplot_w_hue

project_root = '.'
module = os.path.join(project_root, '.')
libraries = '/home/yannick/PycharmProjects/miapy'
plotoutdir = '/hd/BRATS/Brats18/mri_estimator/plots_demogr'

python_path_entries = (project_root, module, *libraries)
for entry in python_path_entries:
    sys.path.insert(0, entry)

import miapy.data.extraction as miapy_extr

import data.storage as data_storage
import analyze.visualizing as visualize
import analyze.scoring as analyze_score

from config import Configuration

import model.alexnet as alexnet


TRAIN_TEST_SPLIT = 0.8

logger = logging.getLogger()

def process_cm(confusion_mat, i=0):
    # i means which class to choose to do one-vs-the-rest calculation
    # rows are actual obs whereas columns are predictions
    TP = confusion_mat[i,i]  # correctly labeled as i
    FP = confusion_mat[:,i].sum() - TP  # incorrectly labeled as i
    FN = confusion_mat[i,:].sum() - TP  # incorrectly labeled as non-i
    TN = confusion_mat.sum().sum() - TP - FP - FN

    return TP, FP, TN, TN

class Trainer:
    def __init__(self, cfg: Configuration, num_workers: int):
        self._cfg = cfg
        self._checkpoint_dir = None
        self._data_store = None
        self._num_workers = num_workers
        self._subjects_train = None
        self._subjects_validate = None
        self._subjects_test = None
        self._checkpoint_last_save = time.time()
        self._checkpoint_idx = -1
        self._xent = 0

    @property
    def checkpoint_dir(self) -> str:
        if self._checkpoint_dir is None:
            tstamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            self._checkpoint_dir = self._cfg.checkpoint_dir.replace('%m', self._cfg.model).replace('%t', tstamp)

        return self._checkpoint_dir

    def assign_subjects(self):
        if self._cfg.subjects_train_val_test_file:
            with open(self._cfg.subjects_train_val_test_file, 'r') as file:
                train, validate, test = file.readline().split(';')
                self._subjects_train = train.split(',')
                self._subjects_validate = validate.split(',')
                self._subjects_test = test.split(',')
        else:
            # split data in train/validation/test
            self._subjects_train, val_test = model_selection.train_test_split(
                [s['subject'] for s in self._data_store.dataset],
                train_size=TRAIN_TEST_SPLIT,
                shuffle=True)
            self._subjects_validate, self._subjects_test = model_selection.train_test_split(val_test, train_size=0.5,
                                                                                            shuffle=True)
        logger.info('train/val/test: %s;%s;%s',
                    ','.join(self._subjects_train), ','.join(self._subjects_validate), ','.join(self._subjects_test))

    @staticmethod
    def set_seed(epoch: int):
        seed = 42 + epoch
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        torch.manual_seed(seed)

    def write_settings(self):
        # write subjects.txt
        with open(os.path.join(self.checkpoint_dir, 'subjects.txt'), 'w') as file:
            file.write('{};{};{}'.format(
                ','.join(self._subjects_train), ','.join(self._subjects_validate), ','.join(self._subjects_test)))

        # write network.py
        net_name = '{}.py'.format(self._cfg.model)
        with open(os.path.join(self.checkpoint_dir, net_name), 'w') as file:
            file.write(inspect.getsource(self.get_python_obj(self._cfg.model)))

        # write config.json
        # make copy of config before making changes
        cfg_tmp = Configuration()
        cfg_tmp.from_dict(self._cfg.to_dict())
        cfg_tmp.checkpoint_dir = self.checkpoint_dir
        cfg_tmp.subjects_train_val_test_file = os.path.join(self.checkpoint_dir, 'subjects.txt')
        # cfg_tmp.z_column_multipliers = self._regression_column_multipliers.tolist()
        cfg_tmp.save(os.path.join(self.checkpoint_dir, 'config.json'))

    # def extract_columns_batch(self, batch):
    #     return np.stack(batch[data_storage.STORE_MORPHOMETRICS], axis=0)[:, self._regression_column_ids]

    def batch_to_feed_dict(self, x_placeholder, y_placeholder, d_placeholder, train_placeholder, batch, is_train) -> dict:
        age = batch['age']
        resectionstatus = batch['resectionstatus']
        #volncr = batch['volncr']
        #voled = batch['voled']
        # volet = batch['volet']
        # etrimwidth = batch['etrimwidth']
        # etgeomhet = batch['etgeomhet']
        # rim_q1_clipped = batch['rim_q1_clipped']
        # rim_q2_clipped = batch['rim_q2_clipped']
        # rim_q3_clipped = batch['rim_q3_clipped']

        demographics = np.stack([age, resectionstatus], axis=1).astype(np.float32)
        # demographics = np.stack([age, resectionstatus, volncr, voled, volet, etrimwidth, etgeomhet, rim_q1_clipped, rim_q2_clipped, rim_q3_clipped], axis=1).astype(np.float32)
        #demographics = np.stack([age, resectionstatus, volncr, voled, volet], axis=1).astype(np.float32)
        #demographics = np.stack([age, resectionstatus], axis=1).astype(np.float32)

        feed_dict = {x_placeholder: np.stack(batch['images'], axis=0).astype(np.float32),
                     d_placeholder: demographics,
                     train_placeholder: is_train}
        if is_train:
            # feed_dict[y_placeholder] = self.extract_columns_batch(batch) / self._regression_column_multipliers
            #feed_dict[y_placeholder] = np.asarray(batch['survclass'])[:, np.newaxis]
            feed_dict[y_placeholder] = np.asarray(batch['survclass'])
            # feed_dict[y_placeholder] = np.asarray(batch['volet'])[:, np.newaxis]

        return feed_dict

    def get_python_obj(self, model_function):
        mod_name, func_name = model_function.rsplit('.', 1)
        mod = importlib.import_module(mod_name)

        return getattr(mod, func_name)

    def get_transform(self):
        transform = None
        if self._cfg.data_augment_transform:
            # e.g. 'data.preprocess.RandomRotateShiftTransform:1:15'
            params = self._cfg.data_augment_transform.split(':')
            do_rotate = True
            shift = 0
            if len(params) > 1:
                do_rotate = int(params[1]) > 0
                shift = int(params[2])

            transform = self.get_python_obj(params[0])(do_rotate, shift)

        return transform

    def evaluate_loss(self, sess, loss, data_loader, x, y, d, is_train):
        sum_cost = 0
        num_samples = 0
        for batch in data_loader:
            print('+', end='', flush=True)
            feed_dict = self.batch_to_feed_dict(x, y, d, is_train, batch, True)
            # print("sum_cost: " + str(sum_cost))
            # print("cost_step: " + str(np.mean(sess.run(loss, feed_dict=feed_dict))))

            loss_eval_step = np.mean(sess.run(loss, feed_dict=feed_dict))

            sum_cost = sum_cost + loss_eval_step*len(batch) #loss_eval_step * len(batch) # sess.run(loss, feed_dict=feed_dict) * len(batch)
            num_samples += len(batch)

        print()
        return sum_cost / num_samples

    def predict(self, sess, net, data_loader, x, y, d, is_train):
        predictions = None
        gt_labels = None
        subject_names = []
        for batch in data_loader:
            print('*', end='', flush=True)
            feed_dict = self.batch_to_feed_dict(x, y, d, is_train, batch, False)
            predicted_labels = np.argmax(sess.run(net, feed_dict=feed_dict)["classes"], axis=1)
            gt = np.argmax(np.asarray(batch['survclass']), axis=1)
            #gt = np.asarray(batch['volet'])
            subject_names.extend(batch['subject'])
            if predictions is None:
                predictions = predicted_labels
                gt_labels = gt
            else:
                predictions = np.concatenate((predictions, predicted_labels))
                gt_labels = np.concatenate((gt_labels, gt))

        print()
        if predictions is None:
            return [], [], []
        else:
            return predictions, gt_labels, subject_names

    def write_results_csv(self, file_name: str, y: np.ndarray, gt: np.ndarray, subjects: typing.List[str]):
        file_path = os.path.join(self.checkpoint_dir, file_name)
        with open(file_path, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            row = ['name', 'survclass', 'survclass_gt']
            # row = ['name', 'volet', 'volet_gt']
            writer.writerow(row)

            #print("y: " +str(y))
            #print("gt: " +str(gt))
            #print("subj: " +str(subjects))
            for idx, subject in enumerate(subjects):
                #row = [subject, y[idx, 0], gt[idx]]
                row = [subject, y[idx], gt[idx]]
                writer.writerow(row)
    def checkpoint_safer(self, sess, saver, epoch_checkpoint, epoch, best_xent_checkpoint, xent=None):
        # persist epoch
        sess.run(epoch_checkpoint.assign(tf.constant(epoch)))

        # checkpoint of best r2 score
        if xent and xent > self._xent:
            # persist best r2 score
            sess.run(best_xent_checkpoint.assign(tf.constant(xent)))
            path = os.path.join(self.checkpoint_dir, 'checkpoint-best-r-2.ckpt')
            logger.info('Saving new best checkpoint %s', path)
            saver.save(sess, path)
            self._xent = xent

        # periodic save
        now = time.time()
        if self._checkpoint_last_save + self._cfg.checkpoint_save_interval < now or epoch >= self._cfg.epochs - 1:
            self._checkpoint_idx = (self._checkpoint_idx + 1) % self._cfg.checkpoint_keep
            path = os.path.join(self.checkpoint_dir, 'checkpoint-{}.ckpt'.format(self._checkpoint_idx))
            logger.info('Saving checkpoint %s', path)
            saver.save(sess, path)
            self._checkpoint_last_save = now

    def train(self):
        self.set_seed(epoch=0)

        transform = self.get_transform()
        self._data_store = data_storage.DataStore(self._cfg.hdf_file, transform)
        dataset = self._data_store.dataset

        self.assign_subjects()

        # prepare loaders and extractors
        training_loader = self._data_store.get_loader(self._cfg.batch_size, self._subjects_train, self._num_workers)
        validation_loader = self._data_store.get_loader(self._cfg.batch_size_eval, self._subjects_validate, self._num_workers)
        testing_loader = self._data_store.get_loader(self._cfg.batch_size_eval, self._subjects_test, self._num_workers)

        # train_extractor = miapy_extr.ComposeExtractor(
        #     [miapy_extr.DataExtractor(categories=('images',)),
        #      miapy_extr.DataExtractor(entire_subject=True, categories=('age', 'resectionstatus', 'volncr', 'voled', 'volet', 'etrimwidth', 'etgeomhet',
        #                                                                'rim_q1_clipped', 'rim_q2_clipped', 'rim_q3_clipped')),
        #      miapy_extr.NamesExtractor(categories=('images',)),
        #      miapy_extr.SubjectExtractor()])

        train_extractor = miapy_extr.ComposeExtractor(
            [miapy_extr.DataExtractor(categories=('images',)),
             #miapy_extr.DataExtractor(entire_subject=True, categories=('age', 'resectionstatus', 'survival', 'volncr', 'voled', 'volet', 'etrimwidth', 'etgeomhet',
             #                                                          'rim_q1_clipped', 'rim_q2_clipped', 'rim_q3_clipped')),
             #miapy_extr.DataExtractor(entire_subject=True, categories=(
             #'age', 'resectionstatus', 'survival', 'volncr', 'voled', 'volet')),
             # miapy_extr.DataExtractor(entire_subject=True, categories=(
             #     'age', 'resectionstatus', 'survclass', 'volncr', 'voled', 'volet', 'etrimwidth', 'etgeomhet',
             #     'rim_q1_clipped', 'rim_q2_clipped', 'rim_q3_clipped')),
             miapy_extr.DataExtractor(entire_subject=True, categories=(
                 'age', 'resectionstatus', 'survclass')),
             miapy_extr.NamesExtractor(categories=('images',)),
             miapy_extr.SubjectExtractor()])

        dataset.set_extractor(train_extractor)

        # # read all labels to calculate multiplier
        # column_values, column_names = self._data_store.get_all_metrics()
        # self._regression_column_ids = np.array([column_names.index(name) for name in self._cfg.regression_columns])
        # self._regression_column_multipliers = np.max(column_values[:, self._regression_column_ids], axis=0)

        # alexnet.SCALE = float(self._data_store.get_intensity_scale_max())

        n_batches = int(np.ceil(len(self._subjects_train) / self._cfg.batch_size))

        logger.info('Net: {}, scale: {}'.format(inspect.getsource(self.get_python_obj(self._cfg.model)), alexnet.SCALE))
        logger.info('Train: {}, Validation: {}, Test: {}'.format(
            len(self._subjects_train), len(self._subjects_validate), len(self._subjects_test)))
        logger.info('n_batches: {}'.format(n_batches))
        logger.info(self._cfg)
        logger.info('checkpoints dir: {}'.format(self.checkpoint_dir))

        shape = dataset.direct_extract(train_extractor, 0)['images'].shape  # extract a subject to obtain shape
        print("Shape: " + str(shape))
        shape_y = dataset.direct_extract(train_extractor, 0)['survclass'].shape  # extract a subject to obtain shape
        #print("Shape_y: " + str(shape_y))

        with tf.Graph().as_default() as graph:
            self.set_seed(epoch=0)  # set again as seed is per graph

            x = tf.placeholder(tf.float32, (None,) + shape, name='x')
            y = tf.placeholder(tf.float32, (None,) + shape_y, name='y')
            d = tf.placeholder(tf.float32, (None, 2), name='d')     # age, resectionstate
            is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

            global_step = tf.train.get_or_create_global_step()
            epoch_checkpoint = tf.Variable(0, name='epoch')
            # best_r2_score_checkpoint = tf.Variable(0.0, name='best_r2_score', dtype=tf.float64)
            best_xent_checkpoint = tf.Variable(0.0, name='best_xent', dtype=tf.float64)

            net = self.get_python_obj(self._cfg.model)({'x': x, 'y': y, 'd': d, 'is_train': is_train})
            #print(net["classes"])
            #print(net["probabilities"])
            #print(net["logits"])
            #logits_test = tf.Session().run(net["logits"])
            #print(logits_test)
            #loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(tf.cast(y, tf.uint8), 3, on_value=1, off_value=0), logits=net["logits"])
            # loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=net)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=net["logits"], weights = 1.0, label_smoothing = 0, scope = None, loss_collection = tf.GraphKeys.LOSSES, reduction = tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
            #tf.losses.softmax_cross_entropy(y, logits=net["logits"])
            learning_rate = None
            optimizer = None
            if self._cfg.optimizer == 'SGD':
                learning_rate = tf.train.exponential_decay(self._cfg.learning_rate, global_step,
                                                           self._cfg.learning_rate_decay_steps,
                                                           self._cfg.learning_rate_decay_rate)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                lr_get = lambda: optimizer._learning_rate.eval()
            elif self._cfg.optimizer == 'Adam':
                learning_rate = tf.Variable(self._cfg.learning_rate, name='lr')
                optimizer = tf.train.AdamOptimizer(learning_rate=self._cfg.learning_rate)
                lr_get = lambda: optimizer._lr

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):   # required for batch_norm
                train_op = optimizer.minimize(loss=loss, global_step=global_step)

            sum_lr = tf.summary.scalar('learning_rate', learning_rate)

            # collect variables to add to summaries (histograms, kernel weight images)
            sum_histograms = []
            sum_kernels = []
            kernel_tensors = []
            kernel_tensor_names = []
            for v in tf.global_variables():
                m = re.match('NET/(layer\d+_\w+)/(kernel|bias):0', v.name)
                if m:
                    var = graph.get_tensor_by_name(v.name)
                    sum_histograms.append(tf.summary.histogram('{}/{}'.format(m.group(1), m.group(2)), var))

                    if m.group(2) == 'kernel' and m.group(1).endswith('conv'):
                        kernel_tensor_names.append(v.name)
                        h, w = visualize.get_grid_size(var.get_shape().as_list())
                        img = tf.Variable(tf.zeros([1, h, w, 1]))
                        kernel_tensors.append(img)
                        sum_kernels.append(tf.summary.image(m.group(1), img))

            summary_writer = tf.summary.FileWriter(self.checkpoint_dir, tf.get_default_graph())

            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(init)

                checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
                if checkpoint:
                    # existing checkpoint found, restoring...
                    if not self._cfg.subjects_train_val_test_file:
                        msg = 'Continue training, but no fixed subject assignments found. ' \
                              'Set subjects_train_val_test_file in config.'
                        logger.error(msg)
                        raise RuntimeError(msg)

                    logger.info('Restoring from ' + checkpoint)
                    saver = tf.train.import_meta_graph(checkpoint + '.meta')
                    saver.restore(sess, checkpoint)
                    sess.run(epoch_checkpoint.assign_add(tf.constant(1)))
                    self._xent = best_xent_checkpoint.eval()
                    #logger.info('Continue with epoch %i (best r2: %f)', epoch_checkpoint.eval(), self._best_r2_score)
                    logger.info('Continue with epoch %i (best xent: %f)', epoch_checkpoint.eval(), self._xent)
                    self._checkpoint_idx = int(re.match('.*/checkpoint-.*(\d+).ckpt', checkpoint).group(1))

                    # load column multipliers from file if available
                    cfg_file = os.path.join(self.checkpoint_dir, 'config.json')
                    # if os.path.exists(cfg_file):
                    #     cfg_tmp = Configuration.load(cfg_file)
                    #     if len(cfg_tmp.z_column_multipliers) > 0:
                    #         self._regression_column_multipliers = np.array(cfg_tmp.z_column_multipliers)
                else:
                    # new training, write config to checkpoint dir
                    self.write_settings()

                summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=global_step.eval())

                for epoch in range(epoch_checkpoint.eval(), self._cfg.epochs + 1):
                    self.set_seed(epoch=epoch)

                    epoch_start_time = timeit.default_timer()
                    loss_sum = 0
                    r2_validation = None

                    # training (enable data augmentation)
                    self._data_store.set_transforms_enabled(True)
                    for batch in training_loader:
                        print('.', end='', flush=True)

                        #print("y type: " + str(y))
                        feed_dict = self.batch_to_feed_dict(x, y, d, is_train, batch, True)

                        # perform training step
                        _, loss_step = sess.run([train_op, loss], feed_dict=feed_dict)
                        #print("loss_sum: " +str(loss_sum))
                        # print("loss_step: " +str(np.mean(loss_step)))
                        loss_sum = loss_sum + (loss_step) * len(batch)

                    # disable transformations (data augmentation) for validation
                    self._data_store.set_transforms_enabled(False)

                    # loss on training data
                    cost_train = loss_sum / len(self._subjects_train)

                    if epoch % self._cfg.log_num_epoch == 0:
                        # loss on validation set
                        #cost_validation = np.mean(self.evaluate_loss(sess, loss, validation_loader, x, y, d, is_train))
                        cost_validation = self.evaluate_loss(sess, loss, validation_loader, x, y, d, is_train)
                        cost_validation_str = '{:.16f}'.format(cost_validation)

                    else:
                        print()
                        cost_validation = None
                        cost_validation_str = '-'

                    logger.info('Epoch:{:4.0f}, Loss train: {:.16f}, Loss validation: {}, lr: {:.16f}, dt={:.1f}'.format(epoch, cost_train, cost_validation_str, lr_get(), timeit.default_timer() - epoch_start_time))

                    # don't write loss for first epoch (usually very high) to avoid scaling issue in graph
                    if epoch > 0:
                        # write summary
                        summary = tf.Summary()
                        summary.value.add(tag='loss_train', simple_value=cost_train)
                        if cost_validation:
                            summary.value.add(tag='loss_validation', simple_value=cost_validation)
                        summary_writer.add_summary(summary, epoch)

                        summary_op = tf.summary.merge([sum_lr])
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, epoch)

                    if epoch % self._cfg.log_eval_num_epoch == 0 and epoch > 0:
                        # calculate and log R2 score on training and validation set
                        eval_start_time = timeit.default_timer()
                        predictions_train, gt_train, subjects_train = self.predict(sess, net, training_loader, x, y, d, is_train)
                        predictions_validation, gt_validation, subjects_validation = self.predict(sess, net, validation_loader, x, y, d, is_train)

                        #predictions_train_class = predictions_train['classes']
                        #predictions_validation_class = predictions_validation['classes']

                        trainCorrectClass = np.equal(gt_train, predictions_train)
                        valCorrectClass = np.equal(gt_validation, predictions_validation)

                        traincorrect = np.sum(trainCorrectClass)/len(trainCorrectClass)
                        valcorrect = np.sum(valCorrectClass)/len(valCorrectClass)

                        # confusion matrix
                        confmat_train = metrics.confusion_matrix(gt_train, predictions_train)
                        # precision_train = np.empty(3)
                        # precision_train[:] = np.nan
                        # recall_train = np.empty(3)
                        # recall_train[:] = np.nan

                        confmat_val = metrics.confusion_matrix(gt_validation, predictions_validation)
                        # precision_val = np.empty(3)
                        # precision_val[:] = np.nan
                        # recall_val = np.empty(3)
                        # recall_val[:] = np.nan
                        # for cl in range(0,3):
                        #     precision_train[cl] = confmat_train[cl][cl] / (confmat_train[0][cl] + confmat_train[1][cl] + confmat_train[2][cl])
                        #     recall_train[cl] = confmat_train[cl][cl] / (confmat_train[cl][0] + confmat_train[cl][1] + confmat_train[cl][2])
                        #     precision_val[cl] = confmat_val[cl][cl] / (confmat_val[0][cl] + confmat_val[1][cl] + confmat_val[2][cl])
                        #     recall_val[cl] = confmat_val[cl][cl] / (confmat_val[cl][0] + confmat_val[cl][1] + confmat_val[cl][2])

                        precision_train = metrics.precision_score(gt_train, predictions_train, average=None)
                        recall_train = metrics.recall_score(gt_train, predictions_train, average=None)

                        precision_val = metrics.precision_score(gt_validation, predictions_validation, average=None)
                        recall_val = metrics.recall_score(gt_validation, predictions_validation, average=None)

                        tp_train = np.zeros(3)
                        fp_train = np.zeros(3)
                        fn_train = np.zeros(3)
                        tn_train = np.zeros(3)
                        tp_val = np.zeros(3)
                        fp_val = np.zeros(3)
                        fn_val = np.zeros(3)
                        tn_val = np.zeros(3)
                        for cl in range(0,3):
                            tp_train[cl], fp_train[cl], fn_train[cl], tn_train[cl] = process_cm(confmat_train, cl)
                            tp_val[cl], fp_val[cl], fn_val[cl], tn_val[cl] = process_cm(confmat_val, cl)

                        specificity_train = tn_train / (tn_train + fp_train)
                        specificity_val = tn_val / (tn_val + fp_val)

                        #logger.info('Epoch:{:4.0f}, R2 train: {:.3f}, R2 validation: {:.8f}, sRho train: {:.3f}, sRho validation: {:.3f}, cl.acc training: {:.1%}, cl.acc validation: {:.1%},  dt={:.1f}s'.format(epoch, r2_train, r2_validation, spearmanR_train, spearmanR_validation, traincorrect, valcorrect, timeit.default_timer() - eval_start_time))
                        logger.info('Epoch:{:4.0f}, cl.acc training: {:.1%}, cl.acc validation: {:.1%},  dt={:.1f}s'.format(epoch, traincorrect, valcorrect, timeit.default_timer() - eval_start_time))

                        # write csv with intermediate results
                        self.write_results_csv('results_train-{0:04.0f}.csv'.format(epoch), predictions_train, gt_train, subjects_train)
                        self.write_results_csv('results_validate-{0:04.0f}.csv'.format(epoch), predictions_validation, gt_validation, subjects_validation)

                        summary = tf.Summary()
                        #summary.value.add(tag='r2_train', simple_value=r2_train)
                        #summary.value.add(tag='r2_validation', simple_value=r2_validation)
                        #summary.value.add(tag='SpearmanRho_train', simple_value=spearmanR_train)
                        #summary.value.add(tag='SpearmanRho_validation', simple_value=spearmanR_validation)
                        summary.value.add(tag='XEnt_train', simple_value=cost_train)
                        summary.value.add(tag='XEnt_validation', simple_value=cost_validation)
                        summary.value.add(tag='Classification_Accuracy_train', simple_value=traincorrect)
                        summary.value.add(tag='Classification_Accuracy_validation', simple_value=valcorrect)
                        summary.value.add(tag='Precision_STS_train', simple_value=precision_train[0])
                        summary.value.add(tag='Precision_STS_val', simple_value=precision_val[0])
                        summary.value.add(tag='Precision_MTS_train', simple_value=precision_train[1])
                        summary.value.add(tag='Precision_MTS_val', simple_value=precision_val[1])
                        summary.value.add(tag='Precision_LTS_train', simple_value=precision_train[2])
                        summary.value.add(tag='Precision_LTS_val', simple_value=precision_val[2])
                        summary.value.add(tag='Recall_STS_train', simple_value=recall_train[0])
                        summary.value.add(tag='Recall_STS_val', simple_value=recall_val[0])
                        summary.value.add(tag='Recall_MTS_train', simple_value=recall_train[1])
                        summary.value.add(tag='Recall_MTS_val', simple_value=recall_val[1])
                        summary.value.add(tag='Recall_LTS_train', simple_value=recall_train[2])
                        summary.value.add(tag='Recall_LTS_val', simple_value=recall_val[2])

                        summary.value.add(tag='Specificity_STS_train', simple_value=specificity_train[0])
                        summary.value.add(tag='Specificity_STS_val', simple_value=specificity_val[0])
                        summary.value.add(tag='Specificity_MTS_train', simple_value=specificity_train[1])
                        summary.value.add(tag='Specificity_MTS_val', simple_value=specificity_val[1])
                        summary.value.add(tag='Specificity_LTS_train', simple_value=specificity_train[2])
                        summary.value.add(tag='Specificity_LTS_val', simple_value=specificity_val[2])

                        summary_writer.add_summary(summary, epoch)

                    if epoch % self._cfg.visualize_layer_num_epoch == 0 and len(sum_histograms) > 0:
                        # write histogram summaries and kernel visualization
                        summary_op = tf.summary.merge(sum_histograms)
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, epoch)

                        for idx, kernel_name in enumerate(kernel_tensor_names):
                            # visualize weights of kernel layer from a middle slice
                            kernel_weights = graph.get_tensor_by_name(kernel_name).eval()
                            # make last axis the first
                            kernel_weights = np.moveaxis(kernel_weights, -1, 0)

                            if len(kernel_weights.shape) > 4:
                                # 3d convolution, remove last (single) channel
                                kernel_weights = kernel_weights[:, :, :, :, 0]

                            if kernel_weights.shape[3] > 1:
                                # multiple channels, take example from middle slide
                                slice_num = int(kernel_weights.shape[3] / 2)
                                kernel_weights = kernel_weights[:, :, :, slice_num:slice_num + 1]

                            grid = visualize.make_grid(kernel_weights)[np.newaxis, :, :, np.newaxis]
                            sess.run(kernel_tensors[idx].assign(grid))

                        summary_op = tf.summary.merge(sum_kernels)
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, epoch)

                    summary_writer.flush()

                    # epoch done
                    self.checkpoint_safer(sess, saver, epoch_checkpoint, epoch, best_xent_checkpoint, cost_validation)

                summary_writer.close()
                logger.info('Training done.')

                if self._xent > 0:
                    # restore checkpoint of best R2 score
                    checkpoint = os.path.join(self.checkpoint_dir, 'checkpoint-best-r-2.ckpt')
                    saver = tf.train.import_meta_graph(checkpoint + '.meta')
                    saver.restore(sess, checkpoint)
                    logger.info('RESTORED best-r-2 checkpoint. Epoch: {}, R2: {:.8f}'.format(
                                epoch_checkpoint.eval(),
                                best_xent_checkpoint.eval()))

                # disable transformations (data augmentation) for test
                self._data_store.set_transforms_enabled(False)

                predictions_train, gt_train, subjects_train = self.predict(sess, net, training_loader, x, y, d, is_train)
                predictions_test, gt_test, subjects_test = self.predict(sess, net, testing_loader, x, y, d, is_train)

                self.write_results_csv('results_train.csv', predictions_train, gt_train, subjects_train)
                self.write_results_csv('results_test.csv', predictions_test, gt_test, subjects_test)

                #predictions_train_class = predictions_train['classes']
                #predictions_test_class = predictions_test['classes']

                trainCorrectClass = np.equal(gt_train, predictions_train)
                testCorrectClass = np.equal(gt_test, predictions_test)

                traincorrect = np.sum(trainCorrectClass) / len(trainCorrectClass)
                testcorrect = np.sum(testCorrectClass) / len(testCorrectClass)

                # confusion matrix
                confmat_train = metrics.confusion_matrix(gt_train, predictions_train)
                confmat_test = metrics.confusion_matrix(gt_test, predictions_test)

                precision_train = metrics.precision_score(gt_train, predictions_train, average=None)
                recall_train = metrics.recall_score(gt_train, predictions_train, average=None)

                precision_test = metrics.precision_score(gt_test, predictions_test, average=None)
                recall_test = metrics.recall_score(gt_test, predictions_test, average=None)

                tp_train = np.zeros(3)
                fp_train = np.zeros(3)
                fn_train = np.zeros(3)
                tn_train = np.zeros(3)
                tp_test = np.zeros(3)
                fp_test = np.zeros(3)
                fn_test = np.zeros(3)
                tn_test = np.zeros(3)
                for cl in range(0, 3):
                    tp_train[cl], fp_train[cl], fn_train[cl], tn_train[cl] = process_cm(confmat_train, cl)
                    tp_test[cl], fp_test[cl], fn_test[cl], tn_test[cl] = process_cm(confmat_test, cl)

                specificity_train = tn_train / (tn_train + fp_train)
                specificity_test = tn_test / (tn_test + fp_test)

                # Note: use scaled metrics for MSE and unscaled (original) for R^2
                if len(gt_train) > 0:
                    #accuracy_train = metrics.mean_squared_error(gt_train, predictions_train)
                    #r2_train = metrics.r2_score(gt_train, predictions_train)

                    #spearmanR_train, _ = scipy.stats.spearmanr(gt_train, predictions_train)
                    trainCorrectClass = np.equal(gt_train, predictions_train)
                    traincorrect = np.sum(trainCorrectClass) / len(trainCorrectClass)


                if len(gt_train) == 0:
                    #accuracy_train = 0
                    #r2_train = 0
                    traincorrect = 0

                if len(gt_test) > 0:
                    #accuracy_test = metrics.mean_squared_error(gt_test, predictions_test)
                    #r2_test = metrics.r2_score(gt_test, predictions_test)
                    #spearmanR_test, _ = scipy.stats.spearmanr(gt_test, predictions_test)
                    testCorrectClass = np.equal(gt_test, predictions_test)
                    testcorrect = np.sum(testCorrectClass) / len(testCorrectClass)

                else:
                    #accuracy_test = 0
                    #r2_test = 0
                    #spearmanR_test = 0
                    testcorrect = 0

                #s = analyze_score.print_summary(subjects_test, ['survival'], predictions_test, gt_test)
                #logger.info('Summary:\n%s-------', s)

                # logger.info('TRAIN: cl.acc: {:.1%}, Precision: {:.4f}:, Recall: {:.4f}, Specificity: {:.4f}'.format(traincorrect, precision_train, recall_train, specificity_train))
                # logger.info('TEST: cl.acc: {:.1%}, Precision: {:.4f}:, Recall: {:.4f}, Specificity: {:.4f}'.format(testcorrect, precision_test, recall_test, specificity_test))

                logger.info('TRAIN: cl.acc: ' + str(traincorrect) + ', Precision: ' +str(precision_train) + ', Recall: ' + str(recall_train) + ', Specificity: ' + str(specificity_train))
                logger.info('TEST: cl.acc: ' + str(testcorrect) + ', Precision: ' +str(precision_test) + ', Recall: ' + str(recall_test) + ', Specificity: ' + str(specificity_test))

