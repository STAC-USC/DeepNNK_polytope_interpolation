__author__ = "shekkizh"
"""Approximate Nearest Neighbor Library"""
import faiss
import os
import numpy as np
import numpy.linalg as LA


class FaissNeighborSearch:
    def __init__(self, d, k, use_gpu=False, add_with_ids=False):
        """
        Initialize the class with the dimension of vectors
        :param k: Number of neighbors to search
        :param d: dimension of the database and query vectors
        """
        self.d = d
        self.index = faiss.IndexFlatL2(self.d)
        self.add_with_ids = add_with_ids
        if self.add_with_ids:
            self.index = faiss.IndexIDMap2(self.index)
        self.use_gpu = use_gpu
        if self.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = "0"
            self.convert_to_gpu()
            # self.index = faiss.GpuIndexFlatL2(res, self.d, flat_config)  # Does brute force neighbor search

        # self.index = faiss.IndexFlatIP(d)
        self.k = k

    def convert_to_gpu(self):
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        self.index = faiss.index_cpu_to_gpu(res, flat_config.device, self.index)

    def load(self, folder_name):
        fname = os.path.join(folder_name, 'train_faiss.index')
        if not os.path.exists(fname):
            return False
        self.index = faiss.read_index(fname)

        if self.use_gpu:
            self.convert_to_gpu()
        return True

    def save(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # with open(os.path.join(folder_name, 'train_data_comments.txt'), 'wb') as f:
        #     f.write(faiss.MatrixStats(self.index.).comments)
        fname = os.path.join(folder_name, 'train_faiss.index')
        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(self.index)
        else:
            index = self.index
        faiss.write_index(index, fname)

    def add_to_database(self, x):
        # x = x/LA.norm(x, axis=1, keepdims=True)
        if self.add_with_ids:
            raise SyntaxError("Missing id argument for data to be added. Use add_to_database_with_ids(...) "
                              "function instead or set add_with_ids flag to False at initialization")
        self.index.add(x)

    def add_to_database_with_ids(self, x, ids):
        if not self.add_with_ids:
            raise SyntaxError("Additional id argument for data to be added. Use add_to_database(...) "
                              "function instead or set add_with_ids flag to True at initialization")
        self.index.add_with_ids(x, ids)

    def search_neighbors(self, q):
        # q = q / LA.norm(q, axis=1, keepdims=True)
        return self.index.search(x=q, k=self.k)

    def get_neighbors(self, indices):
        return np.array([self.index.reconstruct(int(ind)) for ind in indices])

    def remove_neighbors(self, indices):
        self.index.remove_ids(indices)
