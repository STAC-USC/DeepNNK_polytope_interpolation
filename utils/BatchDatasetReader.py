"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc


class BatchDataset:
    def __init__(self, images, labels=None, labels_flag=False):
        """
        Intialize a generic file reader with batching for list of files
        :param file_list: list of files to read - filepaths
        :param image_size: Desired output image size
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        crop = True/ False
        crop_size = #size smaller than image - does central crop of square shape
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        """
        # print("Initializing Batch Dataset Reader...")
        self.images = images.astype(np.float32)
        self.labels_flag = labels_flag
        self.labels = labels
        self.batch_offset = 0
        self.epochs_completed = 0
        self.n_samples = self.images.shape[0]
        self.perm = np.arange(self.n_samples)

    def get_dataset_size(self):
        return self.images.shape[0]

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def permute_data(self):
        np.random.shuffle(self.perm)
        self.images = self.images[self.perm]
        if self.labels_flag:
            self.labels = self.labels[self.perm]

    def _get_batch(self, start, end):
        if not self.labels_flag:
            return self.images[start:end]
        else:
            return self.images[start:end], self.labels[start:end]

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.n_samples:
            # Finished epoch
            self.epochs_completed += 1
            # self.permute_data()
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        return self._get_batch(start, end)

class WeightedBatchDataset(BatchDataset):
    def __init__(self, images, labels=None, labels_flag=False):
        BatchDataset.__init__(self, images=images, labels=labels, labels_flag=labels_flag)
        self.reduced_images = self.images
        self.reduced_labels = self.labels
        # Set weights to uniform sampling
        self.weights = np.ones(self.images.shape[0], dtype=np.float) / self.images.shape[0]

    def set_weights(self, weights):
        if np.any(weights < 0):
            raise EnvironmentError("Sampling weights cannot be negative")
        elif weights.shape[0] != self.images.shape[0]:
            raise EnvironmentError("Size of sampling weights is not equal to dataset size")
        else:
            self.weights = weights
            self.weights /= np.sum(self.weights)

    def reduce_dataset(self, reduction_ratio=0):
        # Choose % of the data for subsequent training
        data_percentage = 1.0 - reduction_ratio
        self.n_samples = int(data_percentage * self.images.shape[0])
        self.perm = np.arange(self.n_samples)
        indices = np.random.choice(np.arange(self.images.shape[0]), size=self.n_samples, replace=False,
                                   p=self.weights)
        self.reduced_images = self.images[indices]
        if self.labels_flag:
            self.reduced_labels = self.labels[indices]


    def permute_data(self):
        np.random.shuffle(self.perm)
        self.reduced_images = self.reduced_images[self.perm]
        if self.labels_flag:
            self.reduced_labels = self.reduced_labels[self.perm]

    def _get_batch(self, start, end):
        if not self.labels_flag:
            return self.reduced_images[start:end]
        else:
            return self.reduced_images[start:end], self.reduced_labels[start:end]

class NeighborBatchDataset(BatchDataset):
    def __init__(self, images, labels, labels_flag=False):
        BatchDataset.__init__(self, images=images, labels=labels, labels_flag=labels_flag)
        self.reconstruction_error = np.zeros(self.n_samples, dtype=np.float32)
        self.node_degree = np.zeros(self.n_samples, dtype=np.float32)
        self.node_neighbors = np.zeros(self.n_samples, dtype=np.int)
        self.neighbors = {}

    def set_metrics(self, node_degree, node_neighbors, reconstruction_error, neighbors):
        self.node_degree = node_degree
        self.node_neighbors = node_neighbors
        self.reconstruction_error = reconstruction_error
        self.neighbors = neighbors

    def permute_data(self):
        # BatchDataset.permute_data(self)
        # self.reconstruction_error = self.reconstruction_error[self.perm]
        # self.node_degree = self.node_degree[self.perm]
        # self.node_neighbors = self.node_neighbors[self.perm]
        pass


    def next_batch(self, batch_size):
        # list indices of selected points
        selected_points = []
        indices = np.arange(self.n_samples)
        if len(self.neighbors.keys()) > 0:
            for itr in range(batch_size):
                # get index of proposed sample
                proposed_point = np.random.choice(indices, 1)[0]
                # remove the neighbors and proposed points from indices
                indices = np.setdiff1d(indices, np.append(self.neighbors[proposed_point], proposed_point),
                                       assume_unique=True)
                selected_points.append(proposed_point)
        else:
            selected_points = np.random.choice(indices, batch_size)

        # for slicing
        selected_points = np.asarray(selected_points)
        return self.images[selected_points], self.labels[selected_points]