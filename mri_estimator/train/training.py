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
import SimpleITK as sitk
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
#import sys
#sys.path.append('/hd/BRATS/Brats18/mri_estimator/scripts')
#from jointplot_w_hue import jointplot_w_hue



project_root = '.'
module = os.path.join(project_root, '.')
#libraries = '/home/yannick/PycharmProjects/miapy'
#plotoutdir = plotdir    #'/home/ubelix/istb/ysuter/BraTS2018/mri_estimator/plots_coleconv_lr1e-4_corr'

#python_path_entries = (project_root, module, *libraries)
#for entry in python_path_entries:
#    sys.path.insert(0, entry)

import miapy.data.extraction as miapy_extr

import data.storage as data_storage
import analyze.visualizing as visualize
import analyze.scoring as analyze_score

from config import Configuration

import model.coleconvnet as convnet


TRAIN_TEST_SPLIT = 0.8

logger = logging.getLogger()

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

def assign_class(survarr):
    dpm = 365.0/12
    classification = survarr.copy()
    classification[classification < 10*dpm] = 0
    classification[(classification >= 10*dpm) & (classification <= 15*dpm)] = 1
    classification[classification > 15*dpm] = 2

    return classification

class Trainer:
    def __init__(self, cfg: Configuration, num_workers: int, plotdir: str):
        self._cfg = cfg
        self._plotdir = plotdir
        self._checkpoint_dir = None
        self._data_store = None
        self._num_workers = num_workers
        self._subjects_train = None
        self._subjects_validate = None
        self._subjects_test = None
        self._checkpoint_last_save = time.time()
        self._checkpoint_idx = -1
        self._best_r2_score = 0

    @property
    def checkpoint_dir(self) -> str:
        if self._checkpoint_dir is None:
            tstamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            self._checkpoint_dir = self._cfg.checkpoint_dir.replace('%m', self._cfg.model).replace('%t', tstamp)

        return self._checkpoint_dir

    def assign_subjects(self):
        if self._cfg.subjects_train_val_test_file:
            with open(self._cfg.subjects_train_val_test_file, 'r') as file:
                train, validate, test = file.readline().rstrip().split(';')
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
        #volet = batch['volet']
        #etrimwidth = batch['etrimwidth']
        #etgeomhet = batch['etgeomhet']
        #rim_q1_clipped = batch['rim_q1_clipped']
        #rim_q2_clipped = batch['rim_q2_clipped']
        #rim_q3_clipped = batch['rim_q3_clipped']

        #demographics = np.stack([age, resectionstatus, volncr, voled, volet, etrimwidth, etgeomhet, rim_q1_clipped, rim_q2_clipped, rim_q3_clipped], axis=1).astype(np.float32)
        #demographics = np.stack([age, resectionstatus, volncr, voled, volet], axis=1).astype(np.float32)
        demographics = np.stack([age, resectionstatus], axis=1).astype(np.float32)

        feed_dict = {x_placeholder: np.stack(batch['images'], axis=0).astype(np.float32),
                     d_placeholder: demographics,
                     train_placeholder: is_train}
        if is_train:
            # feed_dict[y_placeholder] = self.extract_columns_batch(batch) / self._regression_column_multipliers
            feed_dict[y_placeholder] = np.asarray(batch['survival'])[:, np.newaxis]
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

            #print("sum_cost: " + str(sum_cost))
            #print("cost_step: " + str(sess.run(loss, feed_dict=feed_dict) * len(batch)))

            sum_cost = sum_cost + sess.run(loss, feed_dict=feed_dict) * len(batch)
            num_samples += len(batch)

        print()
        return sum_cost / num_samples

    def predict(self, sess, net, data_loader, x, y, d, is_train):
        predictions = None
        gt_labels = None
        # conv3out = None
        fc100 = None
        fc20 = None

        subject_names = []
        for batch in data_loader:
            print('*', end='', flush=True)
            feed_dict = self.batch_to_feed_dict(x, y, d, is_train, batch, False)
            # predicted_labels, conv3out, fc100, fc20 = sess.run(net, feed_dict=feed_dict)
            predicted_labels = sess.run(net, feed_dict=feed_dict)
            # print(predicted_labels)
            gt = np.asarray(batch['survival'])
            #gt = np.asarray(batch['volet'])
            subject_names.extend(batch['subject'])

            #print(predicted_labels)
            if predictions is None:
                predictions = predicted_labels["reg"]
                gt_labels = gt
                # conv3out = predicted_labels["conv3out"]
                fc100 = predicted_labels["fc100"]
                fc20 = predicted_labels["fc20"]
            else:
                predictions = np.concatenate((predictions, predicted_labels["reg"]))
                gt_labels = np.concatenate((gt_labels, gt))
                # print(fc20)
                # print(conv3out.shape)
                # print(predicted_labels["conv3out"].shape)
                # conv3out = np.append(conv3out, predicted_labels["conv3out"], axis=5)
                fc100 = np.append(fc100, predicted_labels["fc100"], axis=0)  # np.concatenate(fc100, predicted_labels[1])
                fc20 = np.append(fc20, predicted_labels["fc20"], axis=0)  # np.concatenate(fc100, predicted_labels[1])
                # fc20 = np.concatenate(fc20, predicted_labels[2])

        # print(fc20)
        if predictions is None:
            print("no prediction!")
            return [], [], [], [], []
        else:
            return predictions, gt_labels, subject_names, fc100, fc20

    def write_results_csv(self, file_name: str, y: np.ndarray, gt: np.ndarray, subjects: typing.List[str]):
        file_path = os.path.join(self.checkpoint_dir, file_name)
        with open(file_path, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            row = ['name', 'survival', 'survival_gt']
            # row = ['name', 'volet', 'volet_gt']
            writer.writerow(row)

            for idx, subject in enumerate(subjects):
                row = [subject, y[idx, 0], gt[idx]]
                writer.writerow(row)

    def write_deepfeat_csv(self, file_name: str, feat: np.ndarray, subjects: typing.List[str]):
        #file_path = os.path.join(self.checkpoint_dir, file_name)
        file_path = os.path.join('/home/yannick/Dropbox/PyRad_Featureextractor/deepfeat2', file_name)
        print(feat)
        print(len(feat))
        with open(file_path, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            row = ['name'] + ['deepfeat' + str(x) for x in range(0, len(feat[0]))]
            # row = ['name', 'volet', 'volet_gt']
            writer.writerow(row)
            #
            # print(feat)
            # featmod = [",".join(str(elem) for elem in feat)]
            # print(featmod)

            for idx, subject in enumerate(subjects):
                #row = ['subject'].append(featmod)
                #print(row)
                row = [subject] + feat[idx].tolist()
                #row = [subject] + featmod
                writer.writerow(row)

    def write_TESTresults_csv(self, file_name: str, y: np.ndarray, subjects: typing.List[str]):
        file_path = os.path.join(self.checkpoint_dir, file_name)
        with open(file_path, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            row = ['name', 'survival']
            writer.writerow(row)
            for idx, subject in enumerate(subjects):
                row = [subject, y[idx, 0]]
                writer.writerow(row)

    def checkpoint_safer(self, sess, saver, epoch_checkpoint, epoch, best_r2_score_checkpoint, r2_score=None):
        # persist epoch
        sess.run(epoch_checkpoint.assign(tf.constant(epoch)))

        # checkpoint of best r2 score
        if r2_score and r2_score > self._best_r2_score:
            # persist best r2 score
            sess.run(best_r2_score_checkpoint.assign(tf.constant(r2_score)))
            path = os.path.join(self.checkpoint_dir, 'checkpoint-best-r-2.ckpt')
            logger.info('Saving new best checkpoint %s', path)
            saver.save(sess, path)
            self._best_r2_score = r2_score

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
             miapy_extr.DataExtractor(ignore_indexing=True, categories=(
                 'age', 'resectionstatus', 'survival')),
             miapy_extr.NamesExtractor(categories=('images',)),
             miapy_extr.SubjectExtractor()])

        dataset.set_extractor(train_extractor)

        # # read all labels to calculate multiplier
        # column_values, column_names = self._data_store.get_all_metrics()
        # self._regression_column_ids = np.array([column_names.index(name) for name in self._cfg.regression_columns])
        # self._regression_column_multipliers = np.max(column_values[:, self._regression_column_ids], axis=0)

        # alexnet.SCALE = float(self._data_store.get_intensity_scale_max())

        n_batches = int(np.ceil(len(self._subjects_train) / self._cfg.batch_size))

        logger.info('Net: {}, scale: {}'.format(inspect.getsource(self.get_python_obj(self._cfg.model)), convnet.SCALE))
        logger.info('Train: {}, Validation: {}, Test: {}'.format(
            len(self._subjects_train), len(self._subjects_validate), len(self._subjects_test)))
        logger.info('n_batches: {}'.format(n_batches))
        logger.info(self._cfg)
        logger.info('checkpoints dir: {}'.format(self.checkpoint_dir))

        shape = dataset.direct_extract(train_extractor, 0)['images'].shape  # extract a subject to obtain shape
        print("Shape: " + str(shape))

        with tf.Graph().as_default() as graph:
            self.set_seed(epoch=0)  # set again as seed is per graph

            x = tf.placeholder(tf.float32, (None,) + shape, name='x')
            y = tf.placeholder(tf.float32, (None, 1), name='y')
            # y = tf.placeholder(tf.float32, (None,) + shape_y, name='y')
            d = tf.placeholder(tf.float32, (None, 2), name='d')     # age, resectionstate
            is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

            global_step = tf.train.get_or_create_global_step()
            epoch_checkpoint = tf.Variable(0, name='epoch')
            best_r2_score_checkpoint = tf.Variable(0.0, name='best_r2_score', dtype=tf.float64)

            # net_full = self.get_python_obj(self._cfg.model)({'x': x, 'y': y, 'd': d, 'is_train': is_train})
            net = self.get_python_obj(self._cfg.model)({'x': x, 'y': y, 'd': d, 'is_train': is_train})     #["reg"]
            print(net)
            print("%%%%%%%%%%% blabla %%%%%%%%%%%%%%%%%")
            print(y.shape)
            print(y)
            loss = tf.losses.mean_squared_error(labels=net["reg"], predictions=y)
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
                    # if m.group(2) == 'kernel' and (m.group(1).find('conv') != -1):
                    # if (m.group(1).find('conv') != -1):
                        kernel_tensor_names.append(v.name)
                        print(kernel_tensor_names)
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
                    self._best_r2_score = best_r2_score_checkpoint.eval()
                    logger.info('Continue with epoch %i (best r2: %f)', epoch_checkpoint.eval(), self._best_r2_score)
                    self._checkpoint_idx = int(re.match('.*/checkpoint-.*(\d+).ckpt', checkpoint).group(1))

                    # # load column multipliers from file if available
                    # cfg_file = os.path.join(self.checkpoint_dir, 'config.json')
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
                    spearmanR_validation = -10

                    # training (enable data augmentation)
                    self._data_store.set_transforms_enabled(True)
                    for batch in training_loader:
                        print('.', end='', flush=True)

                        feed_dict = self.batch_to_feed_dict(x, y, d, is_train, batch, True)

                        # perform training step
                        _, loss_step = sess.run([train_op, loss], feed_dict=feed_dict)
                        loss_sum = loss_sum + loss_step * len(batch)

                    # disable transformations (data augmentation) for validation
                    self._data_store.set_transforms_enabled(False)

                    # loss on training data
                    cost_train = loss_sum / len(self._subjects_train)

                    if epoch % self._cfg.log_num_epoch == 0:
                        # loss on validation set
                        cost_validation = self.evaluate_loss(sess, loss, validation_loader, x, y, d, is_train)
                        cost_validation_str = '{:.16f}'.format(cost_validation)

                    else:
                        print()
                        cost_validation = None
                        cost_validation_str = '-'

                    logger.info('Epoch:{:4.0f}, Loss train: {:.16f}, Loss validation: {}, lr: {:.16f}, dt={:.1f}s'.format(epoch, cost_train, cost_validation_str, lr_get(), timeit.default_timer() - epoch_start_time))

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
                        predictions_train, gt_train, subjects_train, _, _ = self.predict(sess, net, training_loader, x, y, d, is_train)
                        predictions_validation, gt_validation, subjects_validation, _, _ = self.predict(sess, net, validation_loader, x, y, d, is_train)
                        r2_train = metrics.r2_score(gt_train, predictions_train)
                        r2_validation = metrics.r2_score(gt_validation, predictions_validation)
                        spearmanR_train, _ = scipy.stats.spearmanr(gt_train, predictions_train)
                        spearmanR_validation, _ = scipy.stats.spearmanr(gt_validation, predictions_validation)

                        gt_train_class = assign_class(gt_train)
                        predictions_train_class = assign_class(np.squeeze(predictions_train))
                        gt_validation_class = assign_class(gt_validation)
                        predictions_validation_class = assign_class(np.squeeze(predictions_validation))

                        trainCorrectClass = np.equal(gt_train_class, predictions_train_class)
                        valCorrectClass = np.equal(gt_validation_class, predictions_validation_class)

                        traincorrect = np.sum(trainCorrectClass)/len(trainCorrectClass)
                        valcorrect = np.sum(valCorrectClass)/len(valCorrectClass)

                        logger.info('Epoch:{:4.0f}, R2 train: {:.3f}, R2 validation: {:.8f}, sRho train: {:.3f}, sRho validation: {:.3f}, cl.acc training: {:.1%}, cl.acc validation: {:.1%},  dt={:.1f}s'.format(epoch, r2_train, r2_validation, spearmanR_train, spearmanR_validation, traincorrect, valcorrect, timeit.default_timer() - eval_start_time))

                        # write csv with intermediate results
                        self.write_results_csv('results_train-{0:04.0f}.csv'.format(epoch), predictions_train, gt_train, subjects_train)
                        self.write_results_csv('results_validate-{0:04.0f}.csv'.format(epoch), predictions_validation, gt_validation, subjects_validation)

                        summary = tf.Summary()
                        summary.value.add(tag='r2_train', simple_value=r2_train)
                        summary.value.add(tag='r2_validation', simple_value=r2_validation)
                        summary.value.add(tag='SpearmanRho_train', simple_value=spearmanR_train)
                        summary.value.add(tag='SpearmanRho_validation', simple_value=spearmanR_validation)
                        summary.value.add(tag='Classification_Accuracy_train', simple_value=traincorrect)
                        summary.value.add(tag='Classification_Accuracy_validation', simple_value=valcorrect)

                        summary_writer.add_summary(summary, epoch)

                    #if epoch % self._cfg.log_num_epoch == 0 and epoch > 0:
                    if epoch % 1 == 0 and epoch > 0:
                        # plot prediction vs. ground truth on training and validation set
                        plt.ioff()

                        gt_train_class = assign_class(gt_train)
                        predictions_train_class = assign_class(np.squeeze(predictions_train))
                        gt_validation_class = assign_class(gt_validation)
                        predictions_validation_class = assign_class(np.squeeze(predictions_validation))
                        spearmanR_validation, _ = scipy.stats.spearmanr(gt_validation, predictions_validation)
                        trainCorrectClass = np.equal(gt_train_class, predictions_train_class)
                        valCorrectClass = np.equal(gt_validation_class, predictions_validation_class)

                        traincorrect = np.sum(trainCorrectClass)/len(trainCorrectClass)
                        valcorrect = np.sum(valCorrectClass)/len(valCorrectClass)

                        p0 = (sns.jointplot(gt_train, np.squeeze(predictions_train), xlim=(0, np.max(np.append(gt_train, gt_validation))), ylim=(0, np.max(np.append(gt_train, gt_validation))), kind="reg", stat_func=metrics.r2_score).set_axis_labels("GT", "Prediction"))
                        p1 = (sns.jointplot(gt_validation, np.squeeze(predictions_validation), xlim=(0,np.max(np.append(gt_train,gt_validation))), ylim=(0,np.max(np.append(gt_train,gt_validation))), kind="reg", stat_func=metrics.r2_score).set_axis_labels("GT", "Prediction"))

                        p0.ax_joint.set_title('Training \n Accuracy: {:.1%}'.format(traincorrect), pad=-18)
                        p1.ax_joint.set_title('Validation \n Accuracy: {:.1%}'.format(valcorrect), pad=-18)

                        fig = plt.figure(figsize=(16, 8))
                        gs = gridspec.GridSpec(1, 2)

                        mg0 = SeabornFig2Grid(p0, fig, gs[0])
                        mg1 = SeabornFig2Grid(p1, fig, gs[1])

                        gs.tight_layout(fig)

                        # gs.update(top=0.7)
                        plt.suptitle("Epoch " +str(epoch))
                        plt.savefig(os.path.join(self._plotdir, self._cfg.model + "epoch_"+ str(epoch).zfill(4) + ".png"))
                        plt.close(fig)
                        print('Regression plot saved.')

                        # predictions_train, gt_train, subjects_train, fc100_train, fc20_train = self.predict(sess, net,
                        #                                                                                     training_loader,
                        #                                                                                     x, y, d,
                        #                                                                                     is_train)
                        # predictions_test, gt_test, subjects_test, fc100_test, fc20_test = self.predict(sess, net,
                        #                                                                                testing_loader,
                        #                                                                                x, y, d,
                        #                                                                                is_train)
                        #
                        # print(fc100_test[0])
                        # print(fc100_train[0])
                        # print(fc20_test[0])
                        # print(fc20_train[0])
                        #
                        # print("?&&&&&&&&&&&&&&")
                        #
                        # print(len(fc100_test))
                        # print(len(fc100_train))
                        # print(len(fc20_test))
                        # print(len(fc20_train))
                        #
                        # print("?&&&&&&&&&&&&&&")
                        # print(fc100_test[0].shape)
                        # print(fc100_train[0].shape)
                        # print(fc20_test[0].shape)
                        # print(fc20_train[0].shape)
                        #
                        # self.write_results_csv('results_train.csv', predictions_train, gt_train, subjects_train)
                        # self.write_TESTresults_csv('results_test.csv', predictions_test, subjects_test)
                        #
                        # self.write_deepfeat_csv('fc100_test.csv', fc100_test[0], subjects_test)
                        # self.write_deepfeat_csv('fc20_test.csv', fc20_test[0], subjects_test)
                        #
                        # self.write_deepfeat_csv('fc100_train.csv', fc100_train[0], subjects_train)
                        # self.write_deepfeat_csv('fc20_train.csv', fc20_train[0], subjects_train)


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

                        # summary_op = tf.summary.merge(sum_kernels)
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, epoch)

                    summary_writer.flush()

                    # epoch done
                    self.checkpoint_safer(sess, saver, epoch_checkpoint, epoch, best_r2_score_checkpoint, spearmanR_validation)

                summary_writer.close()
                logger.info('Training done.')

                if self._best_r2_score > 0:
                    # restore checkpoint of best R2 score
                    checkpoint = os.path.join(self.checkpoint_dir, 'checkpoint-best-r-2.ckpt')
                    saver = tf.train.import_meta_graph(checkpoint + '.meta')
                    saver.restore(sess, checkpoint)
                    logger.info('RESTORED best-r-2 checkpoint. Epoch: {}, R2: {:.8f}'.format(
                                epoch_checkpoint.eval(),
                                best_r2_score_checkpoint.eval()))

                # disable transformations (data augmentation) for test
                self._data_store.set_transforms_enabled(False)

                predictions_train, gt_train, subjects_train, fc100_train, fc20_train = self.predict(sess, net, training_loader, x, y, d, is_train)
                predictions_val, gt_val, subjects_val, fc100_val, fc20_val = self.predict(sess, net, validation_loader, x, y, d, is_train)
                predictions_test, gt_test, subjects_test, fc100_test, fc20_test = self.predict(sess, net, testing_loader, x, y, d, is_train)

                print(subjects_test)
                print(len(subjects_test))
                print(predictions_test)

                self.write_results_csv('results_train_epoch'+str(epoch_checkpoint.eval())+'.csv', predictions_train, gt_train, subjects_train)
                self.write_TESTresults_csv('results_test_epoch'+str(epoch_checkpoint.eval())+'.csv', predictions_test, subjects_test)

                self.write_deepfeat_csv('fc100_test_epoch'+str(epoch_checkpoint.eval())+'.csv', fc100_test, subjects_test)
                self.write_deepfeat_csv('fc20_test_epoch'+str(epoch_checkpoint.eval())+'.csv', fc20_test, subjects_test)

                self.write_deepfeat_csv('fc100_val_epoch'+str(epoch_checkpoint.eval())+'.csv', fc100_val, subjects_val)
                self.write_deepfeat_csv('fc20_val_epoch'+str(epoch_checkpoint.eval())+'.csv', fc20_val, subjects_val)

                self.write_deepfeat_csv('fc100_train_epoch'+str(epoch_checkpoint.eval())+'.csv', fc100_train, subjects_train)
                self.write_deepfeat_csv('fc20_train_epoch'+str(epoch_checkpoint.eval())+'.csv', fc20_train, subjects_train)

                # # save last conv-layer output as image for inspection
                # convoutimg_test = sitk.GetImageFromArray(np.swapaxes(conv3out_test[0],0,2))
                # convoutimg_train = sitk.GetImageFromArray(np.swapaxes(conv3out_train[0],0,2))
                #
                # sitk.WriteImage(convoutimg_test, os.path.join(self.checkpoint_dir, 'convout3_test.nii.gz'))
                # sitk.WriteImage(convoutimg_train, os.path.join(self.checkpoint_dir, 'convout3_train.nii.gz'))

                gt_train_class = assign_class(gt_train)
                predictions_train_class = assign_class(np.squeeze(predictions_train))
                #gt_test_class = assign_class(gt_test)
                predictions_test_class = assign_class(np.squeeze(predictions_test))

                # Note: use scaled metrics for MSE and unscaled (original) for R^2
                if len(gt_train) > 0:
                    accuracy_train = metrics.mean_squared_error(gt_train, predictions_train)
                    r2_train = metrics.r2_score(gt_train, predictions_train)

                    spearmanR_train, _ = scipy.stats.spearmanr(gt_train, predictions_train)
                    trainCorrectClass = np.equal(gt_train_class, predictions_train_class)
                    traincorrect = np.sum(trainCorrectClass) / len(trainCorrectClass)


                if len(gt_train) == 0:
                    accuracy_train = 0
                    r2_train = 0
                    traincorrect = 0

                #if len(gt_test) > 0:
                    accuracy_test = metrics.mean_squared_error(gt_test, predictions_test)
                    r2_test = metrics.r2_score(gt_test, predictions_test)
                    #spearmanR_test, _ = scipy.stats.spearmanr(gt_test, predictions_test)
                    #testCorrectClass = np.equal(gt_test_class, predictions_test_class)
                    #testcorrect = np.sum(testCorrectClass) / len(testCorrectClass)

                else:
                    accuracy_test = 0
                    r2_test = 0
                    spearmanR_test = 0
                    testcorrect = 0

                #s = analyze_score.print_summary(subjects_test, ['survival'], predictions_test, gt_test)
                #logger.info('Summary:\n%s-------', s)

                logger.info('TRAIN accuracy(mse): {:.8f}, r2: {:.8f}, Spearman Rho: {:.8f}, Classification Accuracy: {:.1%}'.format(accuracy_train, r2_train, spearmanR_train, traincorrect))
                #logger.info('TEST  accuracy(mse): {:.8f}, r2: {:.8f}, Spearman Rho: {:.8f}, Classification Accuracy: {:.1%}'.format(accuracy_test, r2_test, spearmanR_test, testcorrect))


                # visualize.make_kernel_gif(self.checkpoint_dir, kernel_tensor_names)
