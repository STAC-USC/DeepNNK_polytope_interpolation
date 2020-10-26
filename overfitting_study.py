__author__ = "shekkizh"

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# import statsmodels.api as sm
from utils.graph_utils import non_negative_qpsolver, weighted_classifier
import seaborn as sbn
import json
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sbn.set(font_scale=1.5)

def k_param_study(model_path, knn_values, data_type="train"):
    params = json.load(open(os.path.join(model_path, "parameters.json")))
    n_epochs = params["epochs"]
    plt.figure(1, figsize=(10, 6))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    cmap = plt.get_cmap("tab10")
    epoch_range = range(1, n_epochs + 1)
    for i, k in enumerate(knn_values):
        calibrate_results_path = os.path.join(model_path, '%s_calibrate_results_%d' % (data_type, k))
        graph_name = "NNK"
        results = np.load(os.path.join(calibrate_results_path, 'nnk_calibrate_data.npz'))
        nnk_classification_error_rate = results['nnk_classification_error_rate']

        plt.plot(epoch_range, nnk_classification_error_rate, color=cmap(i * 2 + 1), marker='x',
                 linestyle='-', linewidth=1.5, label='%s (k=%d)' % (graph_name, k))
        # #########################################################################################
        graph_name = "KNN"
        results = np.load(os.path.join(calibrate_results_path, 'knn_calibrate_data.npz'))
        knn_classification_error_rate = results['knn_classification_error_rate']

        plt.plot(epoch_range, knn_classification_error_rate, color=cmap(i * 2 + 2), marker='o',
                 linestyle=':', linewidth=1.5, label='%s (k=%d)' % (graph_name, k))
        # #########################################################################################

    model_error_rate = results['model_error_rate']

    results = np.load(os.path.join(model_path, 'SVC_calibrate_results', 'SVC_calibrate_data_CV_5.npz'))
    cv_error = 1 - results['svm_classification_%s_error_rate' % data_type]
    plt.plot(range(1, n_epochs + 1), np.mean(cv_error, axis=1), color=cmap(len(knn_values) * 2 + 1), marker='s',
             linestyle='-.', linewidth=1.5, label='%s (%d-Fold CV)' % ('SVM', 5))

    plt.plot(range(1, n_epochs + 1), model_error_rate, color='k', marker='s',
             linestyle='-', linewidth=1.5, label='Model')

    fig = plt.gcf()
    fig.legend(loc=9, ncol=8, fontsize='small', handletextpad=0.1, labelspacing=0.1)
    plt.grid(linestyle='dashed')
    plt.xlabel("epochs")
    plt.xlim([0.5, 20.5])
    plt.ylim([0.1, 0.7])

    plt.ylabel('Avg. classifier error')
    plt.savefig(os.path.join(model_path, '%s_classifier_error_rate.eps' % data_type),
                bbox_inches='tight')
    # plt.show()
    plt.close()


if __name__ == "__main__":
    model_paths = [
        "logs/conv2d_models_cifar10_basic_under_parameterized",
        "logs/conv2d_models_cifar10_basic",
        "logs/conv2d_models_cifar10_basic_overfit"
    ]
    knn_values = [25, 50, 75]
    data_type = "train"
    for model_path in model_paths:
        k_param_study(model_path, knn_values, "train")