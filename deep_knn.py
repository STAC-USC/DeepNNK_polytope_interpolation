from __future__ import print_function

_author_ = "shekkizh"
"""Deep KNN classification performance"""
# %%
import os
import json
import numpy as np

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow import keras

import utils.tensorflow_utils as tf_utils
from utils.ann_utils import FaissNeighborSearch as ANN
from utils.BatchDatasetReader import BatchDataset

from utils.non_neg_qpsolver import non_negative_qpsolver
from utils.graph_utils import majority_vote_classifier, weighted_classifier

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sbn
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sbn.set(font_scale=1.5)

MEANS = [0.49139968, 0.48215841, 0.44653091]
STDS = [0.24703223, 0.24348513, 0.26158784]


class Deep_KNN:
    def __init__(self, config=None, flags=None):
        self.flags = flags
        self.DATASET = self.flags.dataset.lower()
        self.IMAGE_SHAPE = [0, 0, 0]
        self.class_labels = None
        self.num_classes = 0
        self.train_dataset = BatchDataset(images=np.empty([1]))
        self.validation_dataset = BatchDataset(images=np.empty([1]))
        self.test_dataset = BatchDataset(images=np.empty([1]))
        self.samples_per_batch = self.flags.processing_size
        self.x_entropy = tf.constant(0, dtype=tf.float32)
        self.loss = tf.constant(0, dtype=tf.float32)
        self.accuracy_op = None
        self.logits = None
        self.pred = None
        self.train_op = None
        self.saver = None
        self.model_output_folder = self.setup_output_dir()

        print("Reading dataset %s ..." % self.DATASET)
        self.read_dataset()
        print(self.train_dataset.get_dataset_size(), 'train samples')
        print(self.validation_dataset.get_dataset_size(), 'validation samples')
        print(self.test_dataset.get_dataset_size(), 'test samples')

        self.images = tf.placeholder(tf.float32, shape=[None] + self.IMAGE_SHAPE, name="input_images")
        self.labels = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="input_labels")
        self.keep_prob = tf.placeholder_with_default(1.0, shape=[], name="keep_prob")
        self.is_training = tf.placeholder_with_default(False, shape=[], name="is_training")

        self.net = {}
        model_settings_file = os.path.join(self.model_output_folder, 'parameters.json')
        with open(model_settings_file, 'w') as f:
            json.dump(self.flags.flag_values_dict(), f)

        print("Setting up model architecture ...")
        self.build_model(tf_utils.augment_data(self.images, self.flags.regularize, self.is_training))

        print("Setting up session ...")
        self.sess = tf.Session(graph=tf.get_default_graph(), config=config)
        self.sess.run(tf.global_variables_initializer())
        self.load()

    def setup_output_dir(self):
        output_folder_name = f"conv2d_models_{self.DATASET}_layer_size_{self.flags.layer_size}_regularized_{self.flags.regularize}"
        model_output_folder = os.path.join(self.flags.logs_dir, output_folder_name)
        if not os.path.exists(model_output_folder):
            os.makedirs(model_output_folder)
        return model_output_folder

    def read_dataset(self):
        if self.DATASET == 'cifar100':
            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
            self.IMAGE_SHAPE = [32, 32, 3]

        elif self.DATASET == 'cifar10':
            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
            self.IMAGE_SHAPE = [32, 32, 3]
        elif self.DATASET == 'mnist':
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
            x_train = x_train.reshape(60000, 28, 28, 1)
            x_test = x_test.reshape(10000, 28, 28, 1)
            self.IMAGE_SHAPE = [28, 28, 1]
        else:
            raise Exception("Dataset not found")

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        self.train_mean = 0
        x_train -= self.train_mean
        x_test -= self.train_mean
        self.train_std = 1

        # convert class vectors to binary class matrices
        self.class_labels = tf_utils.get_class_names(self.DATASET)
        self.num_classes = len(self.class_labels)
        # y_train_scalar = np.copy(y_train)  # memorize scalar values
        # y_test_scalar = np.copy(y_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        x_train, y_train = tf_utils.permute_data(x_train, y_train)
        percentage_training = int(x_train.shape[0] * self.flags.labelled_percent)
        self.train_dataset = self.train_BatchDataset(images=x_train[:percentage_training],
                                                     labels=y_train[:percentage_training])

        percentage_validation = int(x_train.shape[0] * self.flags.validation_percent)
        start_index = percentage_training
        end_index = start_index + percentage_validation
        if end_index <= x_train.shape[0]:
            self.validation_dataset = BatchDataset(images=x_train[start_index:end_index],
                                                   labels=y_train[start_index:end_index],
                                                   labels_flag=True)

        self.test_dataset = BatchDataset(images=x_test, labels=y_test, labels_flag=True)

    def train_BatchDataset(self, images, labels):
        return BatchDataset(images=images, labels=labels, labels_flag=True)

    def load(self):
        """Restores parameters of model from `model_in_file`."""
        self.saver = tf.train.Saver(max_to_keep=20)
        ckpt = tf.train.get_checkpoint_state(self.model_output_folder)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored ... %s" % ckpt.model_checkpoint_path)

    def dropout_layer(self, x):
        if self.flags.regularize:
            return tf.nn.dropout(x, self.keep_prob)
        else:
            return x

    def network_architecture(self, input_data, scope_name="network", scope_reuse=False):
        activation_dict = {}
        layer_size = self.flags.layer_size
        with tf.variable_scope(scope_name, reuse=scope_reuse):
            W = tf_utils.weight_variable([3, 3, self.IMAGE_SHAPE[2], layer_size], name="W_conv0")
            b = tf_utils.bias_variable([layer_size], name="b_conv0")
            activation_dict[0] = self.dropout_layer(tf.nn.relu(tf_utils.conv2d_basic(input_data, W, b)))
            for ii in range(1, self.flags.n_layers + 1):
                W = tf_utils.weight_variable([3, 3, layer_size, layer_size], name="W_conv" + str(ii))
                b = tf_utils.bias_variable([layer_size], name="b_conv" + str(ii))
                activation = tf.nn.relu(tf_utils.conv2d_basic(activation_dict[ii - 1], W, b))
                activation_dict[ii] = self.dropout_layer(activation)
                if (ii + 1) % 2 == 0:  # Pool after every 2 layers of convolution
                    activation_dict[ii] = tf_utils.max_pool_2x2(activation_dict[ii])

        return activation_dict

    def build_model(self, input_data):
        scope_name = "network"
        self.net = self.network_architecture(input_data, scope_name=scope_name, scope_reuse=False)
        # No. of Pool layers is (self.flags.n_layers+1)/2. Shape reduction is on two axis
        shape_reduction = 2 ** (self.flags.n_layers + 1)
        net_size = int(self.flags.layer_size * self.IMAGE_SHAPE[0] * self.IMAGE_SHAPE[1] / shape_reduction)
        net_flatten = tf.reshape(self.net[self.flags.n_layers], [-1, net_size])

        W_fc1 = tf_utils.weight_variable([net_size, self.num_classes], name="W_fc1")
        b_fc1 = tf_utils.bias_variable([self.num_classes], name="b_fc1")
        self.logits = tf.matmul(net_flatten, W_fc1) + b_fc1
        self.pred = tf.nn.softmax(self.logits)

        self.x_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        tf.summary.scalar("X-entropy", self.x_entropy)

        self.loss = self.x_entropy

        self.accuracy_op = tf_utils.model_accuracy(pred=self.pred, labels=self.labels)

        train_variables = tf.trainable_variables()
        # for v in train_variables:
        #     tf_utils.add_to_regularization_and_summary(var=v)
        self.train_op = tf_utils.train(self.loss, train_variables, self.flags.learning_rate)

    def update_train_dataset(self, epoch):
        self.train_dataset.reset_batch_offset()
        self.train_dataset.permute_data()

    def fit(self):
        summary_writer = tf.summary.FileWriter(self.model_output_folder, graph=self.sess.graph,
                                               session=self.sess)
        summary_op = tf.summary.merge_all()

        for epoch in range(self.flags.epochs):
            self.update_train_dataset(epoch)
            itr_per_epoch = int(self.train_dataset.n_samples / self.flags.batch_size)
            for itr in range(itr_per_epoch):
                batch_images, batch_labels = self.train_dataset.next_batch(batch_size=self.flags.batch_size)
                feed_dict = {self.images: batch_images, self.labels: batch_labels, self.keep_prob: 0.9,
                             self.is_training: True}
                #
                self.sess.run(self.train_op, feed_dict=feed_dict)
                if itr % 100 == 0:
                    # feed_dict[self.keep_prob] = 1.0
                    tot_loss, xentropy_loss, summary_str = self.sess.run([self.loss, self.x_entropy, summary_op],
                                                                         feed_dict=feed_dict)
                    print("Epoch: %d , Itr:%d , Loss : %g , X-Entropy Loss: %g" % (epoch, itr, tot_loss, xentropy_loss))
                    summary_writer.add_summary(summary_str, global_step=epoch * itr_per_epoch + itr)
            if self.flags.validation_percent > 0:
                validation_loss, validation_accuracy = self.get_performance(self.validation_dataset)
                print("Validation Data results - X-Entropy: %g, Accuracy %g" % (
                    validation_loss, np.mean(validation_accuracy)))
            self.saver.save(self.sess, self.model_output_folder + '/model.ckpt', epoch * itr_per_epoch)

    def get_performance(self, dataset):
        dataset_size = dataset.get_dataset_size()
        n_batches = dataset_size // self.samples_per_batch
        last_batch = dataset_size % self.samples_per_batch
        if n_batches == 0:
            return 0, 0
        loss = np.zeros(n_batches, dtype=np.float)
        accuracy = np.zeros(n_batches, dtype=np.float)
        start_idx = 0
        for itr in trange(n_batches, desc="Processing batches for performance"):
            end_idx = start_idx + self.samples_per_batch
            feed_dict = {self.images: dataset.images[start_idx:end_idx],
                         self.labels: dataset.labels[start_idx:end_idx]}  # , self.keep_prob: 0.1
            loss[itr], acc = self.sess.run([self.x_entropy, self.accuracy_op], feed_dict=feed_dict)
            accuracy[itr] = np.mean(acc)
            start_idx = end_idx
        return np.mean(loss), np.mean(accuracy)

    def test(self):
        test_loss, test_accuracy = self.get_performance(self.test_dataset)
        print("Test Data results - X-Entropy: %g, Accuracy %g" % (test_loss, test_accuracy))
        train_loss, train_accuracy = self.get_performance(self.train_dataset)
        print("Train Data results - X-Entropy: %g, Accuracy %g" % (train_loss, train_accuracy))
        validation_loss, validation_accuracy = self.get_performance(self.validation_dataset)
        print("Validation Data results - X-Entropy: %g, Accuracy %g" % (validation_loss, validation_accuracy))

        output_results_file = os.path.join(self.model_output_folder, 'results.json')
        with open(output_results_file, 'w') as f:
            results = {
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'validation_loss': validation_loss,
                'validation_accuracy': validation_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }
            json.dump(results, f)

    def train_neighbor_search(self, layer, knn_param=1, use_gpu=False, folder_prefix="", save_ann=True):
        d = tf_utils.get_tensor_size(self.net[layer])
        neighbor_search = ANN(d, knn_param, use_gpu=use_gpu)
        train_neighbor_folder = os.path.join(self.model_output_folder, folder_prefix)
        if neighbor_search.load(train_neighbor_folder):
            return neighbor_search

        n_batches = int(self.train_dataset.get_dataset_size() / self.samples_per_batch)
        for itr in trange(n_batches, desc="Adding train data to ANN"):
            indices = range(itr * self.samples_per_batch, (itr + 1) * self.samples_per_batch)
            batch_images = self.train_dataset.images[indices]
            activation = self.sess.run(self.net[layer], feed_dict={self.images: batch_images})
            neighbor_search.add_to_database(x=np.reshape(activation, [self.samples_per_batch, d]))
        if save_ann:
            neighbor_search.save(train_neighbor_folder)
        return neighbor_search

    def calibrate_data(self, folder_prefix=""):
        data_type = self.flags.data_type
        knn_param = self.flags.knn_param
        if data_type == "train":
            test_dataset = self.train_dataset
            knn_param += 1  # Need to search for extra neighbor to avoid self
        elif data_type == "test":
            test_dataset = self.test_dataset
        else:
            raise EnvironmentError("Unknown calibration save data type: %s" % data_type)

        calibrate_results_path = os.path.join(self.model_output_folder,
                                              "%s_calibrate_results_%d/" % (data_type, self.flags.knn_param),
                                              folder_prefix)
        if not os.path.exists(calibrate_results_path):
            os.makedirs(calibrate_results_path)
        n_batches = int(test_dataset.get_dataset_size() / self.samples_per_batch)
        n_epochs = self.flags.epochs

        ckpt = tf.train.get_checkpoint_state(self.model_output_folder)
        ckpt_paths = ckpt.all_model_checkpoint_paths
        knn_layers = sorted(map(int, self.flags.knn_layers.split(",")))
        for layer_itr in range(len(knn_layers)):
            layer = knn_layers[layer_itr]

            knn_classification_error_rate = np.zeros(n_epochs, dtype=np.float)
            knn_classification_error_rate2 = np.zeros(n_epochs, dtype=np.float)

            nnk_error_rate = np.zeros(n_epochs, dtype=np.float)
            nnk_classification_error_rate = np.zeros(n_epochs, dtype=np.float)
            nnk_classification_error_rate2 = np.zeros(n_epochs, dtype=np.float)

            model_error_rate = np.zeros(n_epochs, dtype=np.float)
            node_degree = np.zeros((n_epochs, n_batches, self.samples_per_batch), dtype=np.float)
            node_neighbors = np.zeros((n_epochs, n_batches, self.samples_per_batch), dtype=np.float)
            for epoch_itr in range(len(ckpt_paths)):
                self.saver.restore(self.sess, ckpt_paths[epoch_itr])
                neighbor_search = self.train_neighbor_search(layer, knn_param=knn_param,
                                                             folder_prefix="%d" % epoch_itr)

                knn_prediction_error = np.zeros((n_batches, self.samples_per_batch), dtype=np.float)
                knn_prediction_error2 = np.zeros((n_batches, self.samples_per_batch), dtype=np.float)

                nnk_reconstruction_error = np.zeros((n_batches, self.samples_per_batch), dtype=np.float)
                nnk_prediction_error = np.zeros((n_batches, self.samples_per_batch), dtype=np.float)
                nnk_prediction_error2 = np.zeros((n_batches, self.samples_per_batch), dtype=np.float)
                model_error = np.zeros((n_batches, self.samples_per_batch), dtype=np.float)

                for itr in trange(n_batches, desc="Querying for neighbors of test data samples"):
                    batch_images, batch_labels = test_dataset.next_batch(self.samples_per_batch)
                    activation, accuracy = self.sess.run([self.net[layer], self.accuracy_op],
                                                         feed_dict={self.images: batch_images,
                                                                    self.labels: batch_labels})
                    queries = np.reshape(activation, [self.samples_per_batch, neighbor_search.d])
                    y_train = np.zeros((self.samples_per_batch, self.flags.knn_param, self.num_classes),
                                       dtype=np.float)
                    dist, ind = neighbor_search.search_neighbors(q=queries)
                    if data_type == "train":
                        D = dist[:, 1:]
                        I = ind[:, 1:]
                    else:
                        D = dist
                        I = ind
                    D_knn = np.zeros_like(D)
                    reconstruction_error = np.zeros(self.samples_per_batch, dtype=np.float)
                    for ii in range(self.samples_per_batch):
                        x_train = neighbor_search.get_neighbors(I[ii, :])
                        y_train[ii, :, :] = self.train_dataset.labels[I[ii, :]]
                        x_train = x_train / np.linalg.norm(x_train, axis=1, keepdims=True)
                        x_test = queries[ii, :]
                        x_test = x_test / np.linalg.norm(x_test)
                        g_i = 0.5 + np.dot(x_train, x_test) / 2
                        D_knn[ii, :] = g_i
                        G_i = 0.5 + np.dot(x_train, x_train.T) / 2
                        x_opt, check = non_negative_qpsolver(G_i, g_i, g_i, self.flags.edge_threshold)
                        reconstruction_error[ii] = 1 - 2 * np.dot(g_i, x_opt) + np.dot(x_opt, np.dot(G_i, x_opt))
                        D[ii, :] = x_opt
                        node_degree[epoch_itr, itr, ii] = np.sum(x_opt)
                        node_neighbors[epoch_itr, itr, ii] = np.count_nonzero(x_opt)

                    nnk_prediction_error[itr] = majority_vote_classifier(D, y_train, batch_labels)
                    nnk_prediction_error2[itr] = weighted_classifier(D, y_train, batch_labels)

                    knn_prediction_error[itr] = majority_vote_classifier(D_knn, y_train, batch_labels)
                    knn_prediction_error2[itr] = weighted_classifier(D_knn, y_train, batch_labels)
                    nnk_reconstruction_error[itr] = reconstruction_error
                    model_error[itr] = 1 - accuracy

                knn_classification_error_rate[epoch_itr] = np.mean(knn_prediction_error)
                knn_classification_error_rate2[epoch_itr] = np.mean(knn_prediction_error2)

                nnk_error_rate[epoch_itr] = np.mean(nnk_reconstruction_error)
                nnk_classification_error_rate[epoch_itr] = np.mean(nnk_prediction_error)
                nnk_classification_error_rate2[epoch_itr] = np.mean(nnk_prediction_error2)
                model_error_rate[epoch_itr] = np.mean(model_error)

            np.savez_compressed(os.path.join(calibrate_results_path, 'nnk_calibrate_data.npz'),
                                nnk_error_rate=nnk_error_rate,
                                nnk_classification_error_rate=nnk_classification_error_rate,
                                nnk_classification_error_rate2=nnk_classification_error_rate2,
                                model_error_rate=model_error_rate, node_degree=node_degree,
                                node_neighbors=node_neighbors)
            np.savez_compressed(os.path.join(calibrate_results_path, 'knn_calibrate_data.npz'),
                                knn_classification_error_rate=knn_classification_error_rate,
                                knn_classification_error_rate2=knn_classification_error_rate2,
                                model_error_rate=model_error_rate)

    def plot_neighbors(self, indices):
        data_type = self.flags.data_type
        knn_param = self.flags.knn_param
        if data_type == "train":
            test_dataset = self.train_dataset
            knn_param += 1  # Need to search for extra neighbor to avoid self
        elif data_type == "test":
            test_dataset = self.test_dataset
        else:
            raise EnvironmentError("Unknown calibration save data type: %s" % data_type)
        plot_results_path = os.path.join(self.model_output_folder, "plot_results/")
        if not os.path.exists(plot_results_path):
            os.makedirs(plot_results_path)
        knn_layers = sorted(map(int, self.flags.knn_layers.split(",")))
        layer = knn_layers[-1]
        neighbor_search = self.train_neighbor_search(layer, knn_param=knn_param,
                                                     folder_prefix="%d" % (self.flags.epochs - 1))

        batch_images = test_dataset.images[indices]
        batch_labels = np.argmax(test_dataset.labels[indices], axis=1)
        activation = self.sess.run(self.net[layer], feed_dict={self.images: batch_images})
        queries = np.reshape(activation, [len(indices), neighbor_search.d])
        D, I = neighbor_search.search_neighbors(q=queries)
        D = D[:, 1:]
        I = I[:, 1:]
        for ii in trange(len(indices), desc="Processing test input for plot", leave=True, position=0):
            x_train = neighbor_search.get_neighbors(I[ii, :])
            y_train = np.argmax(self.train_dataset.labels[I[ii, :]], axis=1)
            x_train_images = self.train_dataset.images[I[ii, :]]
            x_train = x_train / np.linalg.norm(x_train, axis=1, keepdims=True)
            x_test = queries[ii, :]
            x_test = x_test / np.linalg.norm(x_test)
            g_i = 0.5 + np.dot(x_train, x_test) / 2
            G_i = 0.5 + np.dot(x_train, x_train.T) / 2
            x_opt, check = non_negative_qpsolver(G_i, g_i, g_i, self.flags.edge_threshold)
            non_zero = np.nonzero(x_opt)[0]
            no_neighbors = len(non_zero)
            W = x_opt / np.sum(x_opt)
            predicted_label = np.argmax(np.sum(np.expand_dims(W, axis=1) * self.train_dataset.labels[I[ii, :]], axis=0))

            fontdict = {'fontsize': 6}
            images_per_row = 7
            plt_rows = np.ceil((1. + no_neighbors) / images_per_row).astype(np.int)
            plt.figure(figsize=(6.4, plt_rows * 1.2))
            ax = plt.subplot(plt_rows, images_per_row, 1)
            plt.imshow(batch_images[ii] * self.train_std + self.train_mean, aspect='equal')
            ax.set_axis_off()
            ax.set_title("%s, %s " % (self.class_labels[batch_labels[ii]], self.class_labels[predicted_label]),
                         fontdict=fontdict)
            sorted_index = np.argsort(W)[::-1]
            W = W[sorted_index]
            x_train_images = x_train_images[sorted_index]
            y_train = y_train[sorted_index]
            for neighbor_itr in range(no_neighbors):
                ax = plt.subplot(plt_rows, images_per_row, 1 + 1 + neighbor_itr)  # ((neighbor_itr+1)//images_per_row)
                plt.imshow(x_train_images[neighbor_itr] * self.train_std + self.train_mean, aspect='equal')
                ax.set_axis_off()
                ax.set_title("%s, %0.2e " % (self.class_labels[y_train[neighbor_itr]], W[neighbor_itr]),
                             fontdict=fontdict)
            plt.savefig(
                os.path.join(plot_results_path, "%s_%d_neighbors_%d.eps" % (data_type, indices[ii], no_neighbors)),
                bbox_inches='tight')
            plt.close()

    def get_activations(self, data_type, layer, folder_prefix=""):
        if data_type == "train":
            dataset = self.train_dataset
        elif data_type == "test":
            dataset = self.test_dataset
        else:
            raise EnvironmentError("Unknown calibration save data type: %s" % data_type)
        fname = os.path.join(self.model_output_folder, folder_prefix,
                             '%s_activations_layer_%d.npz' % (data_type, layer))
        if os.path.exists(fname):
            data = np.load(fname)
            return data['X'], data['y']

        d = tf_utils.get_tensor_size(self.net[layer])
        samples_per_batch = self.samples_per_batch
        n_batches = int(dataset.get_dataset_size() / samples_per_batch)
        X = np.zeros((n_batches * samples_per_batch, d), dtype=np.float)
        y = np.zeros((n_batches * samples_per_batch, self.num_classes), dtype=np.float)
        for itr in range(n_batches):
            batch_images, batch_labels = dataset.next_batch(samples_per_batch)
            activation = self.sess.run(self.net[layer], feed_dict={self.images: batch_images})
            X[itr * samples_per_batch:(itr + 1) * samples_per_batch] = np.reshape(activation,
                                                                                  [samples_per_batch, d])
            y[itr * samples_per_batch:(itr + 1) * samples_per_batch] = batch_labels
        np.savez_compressed(fname, X=X, y=y, layer=layer, data_type=data_type)
        return X, y

    def svm_cv_calibrate(self, folder_prefix=""):
        from sklearn.model_selection import KFold
        from sklearn.svm import LinearSVC
        n_cv = self.flags.cross_validation
        kf = KFold(n_splits=n_cv)

        calibrate_results_path = os.path.join(self.model_output_folder,
                                              "SVC_calibrate_results/",
                                              folder_prefix)
        if not os.path.exists(calibrate_results_path):
            os.makedirs(calibrate_results_path)

        n_epochs = self.flags.epochs
        ckpt = tf.train.get_checkpoint_state(self.model_output_folder)
        ckpt_paths = ckpt.all_model_checkpoint_paths
        knn_layers = sorted(map(int, self.flags.knn_layers.split(",")))
        for layer_itr in range(len(knn_layers)):
            layer = knn_layers[layer_itr]

            svm_classification_train_error_rate = np.zeros((n_epochs, n_cv), dtype=np.float)
            svm_classification_test_error_rate = np.zeros((n_epochs, n_cv), dtype=np.float)

            for epoch_itr in range(len(ckpt_paths)):
                self.saver.restore(self.sess, ckpt_paths[epoch_itr])
                X_train, y_train = self.get_activations("train", layer, folder_prefix='%d' % epoch_itr)
                y_train = np.argmax(y_train, axis=1)
                X_test, y_test = self.get_activations("test", layer, folder_prefix='%d' % epoch_itr)
                y_test = np.argmax(y_test, axis=1)
                cv_index = 0
                for train_index, valid_index in kf.split(X_train, y_train):
                    print('Training and testing for split %d' % cv_index)
                    clf = LinearSVC(C=1000).fit(X_train[train_index], y_train[train_index])
                    svm_classification_train_error_rate[epoch_itr, cv_index] = clf.score(X_train[valid_index],
                                                                                         y_train[valid_index])
                    svm_classification_test_error_rate[epoch_itr, cv_index] = clf.score(X_test, y_test)
                    cv_index += 1

            np.savez_compressed(os.path.join(calibrate_results_path, 'SVC_calibrate_data_CV_%d.npz' % n_cv),
                                n_cv=n_cv, svm_classification_train_error_rate=svm_classification_train_error_rate,
                                svm_classification_test_error_rate=svm_classification_test_error_rate)
