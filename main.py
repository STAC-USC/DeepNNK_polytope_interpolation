__author__ = "shekkizh"

import os, random
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from absl import flags, app
from deep_knn import Deep_KNN

tf.logging.set_verbosity('ERROR')
# %% Setting seed for reproducibility
seed_value = 4629
os.environ["PYTHONHASHSEED"] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_random_seed(seed_value)
FLAGS = flags.FLAGS

flags.DEFINE_string("experiment", "sampling", "Type of experiment to run: neighbor/sampling")
# %% Model architecture and folders
flags.DEFINE_string("mode", "train", "train/ test/calibrate /plot/smoothness Mode")
flags.DEFINE_bool("use_gpu", True, "Flag to set usage of GPU by Tensorflow only")
flags.DEFINE_string("dataset", "cifar10", "Dataset to use (mnist, cifar10)")
flags.DEFINE_bool("regularize", False, "Flag to augment dataset, use dropout")
flags.DEFINE_string("logs_dir", "logs/", "Path to logs dir")

# %% Model hyperparameters
flags.DEFINE_integer("batch_size", 50, "Train batch size")
flags.DEFINE_integer("epochs", 20, "Number of epochs to train")
flags.DEFINE_integer("n_layers", 5, "Size of Neural Network")
flags.DEFINE_integer("layer_size", 32, "Size of each layer")
# flags.DEFINE_string("data_dir", "data/", "Path to data dir")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
flags.DEFINE_float("validation_percent", 0., "Percentage of labelled data to use for validation")

# %% Graph related parameters
flags.DEFINE_float("labelled_percent", 1.0, "Percentage of labelled data to use for training")
flags.DEFINE_integer("knn_param", 50, "Max Number of neighbors to use")
flags.DEFINE_string("knn_layers", "5",
                    "Comma separated values corresponding to layers where KNN is performed. For e.g 1,2")
flags.DEFINE_float("edge_threshold", 1e-10, "Threshold value for edge weights")
flags.DEFINE_integer("processing_size", 100, "Number of samples to process at a time while calibrating")
# %% Calibrate related parameters
flags.DEFINE_integer("cross_validation", 5, "cross validation fold for calibrating using linear SVM")

# %%
def main(arg=None):
    session_config = None
    if not FLAGS.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    experiment = FLAGS.experiment
    if experiment == "neighbor":
        model = Deep_KNN(config=session_config, flags=FLAGS)
    else:
        raise EnvironmentError("unknown experiment: %s" % experiment)

    mode = FLAGS.mode
    if mode == "train":
        model.fit()
    elif mode == "test":
        model.test()
    elif mode == "calibrate":
        model.calibrate_data()
        model.svm_cv_calibrate()
    elif mode == "plot":
        model.plot_neighbors([2595, 3745, 374])
    else:
        raise EnvironmentError("Unknown processing mode %s" % mode)


if __name__ == "__main__":
    app.run(main)
