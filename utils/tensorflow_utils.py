__author__ = "shekkizh"
"""Tensorflow utility functions"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os, sys
from six.moves import urllib
import tarfile
import zipfile
import scipy.io
import scipy.misc as misc


# %% Dataset related utils

def get_model_data(dir_path):
    model_url = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
    maybe_download_and_extract(dir_path, model_url)
    filename = model_url.split("/")[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found!")
    data = scipy.io.loadmat(filepath)
    return data


def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)


def get_class_names(dataset="mnist"):
    class_names = {
        "mnist": range(0, 10),
        "cifar10": ['plane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        "cifar100": [
            'apple',  # id 0
            'aquarium_fish',
            'baby',
            'bear',
            'beaver',
            'bed',
            'bee',
            'beetle',
            'bicycle',
            'bottle',
            'bowl',
            'boy',
            'bridge',
            'bus',
            'butterfly',
            'camel',
            'can',
            'castle',
            'caterpillar',
            'cattle',
            'chair',
            'chimpanzee',
            'clock',
            'cloud',
            'cockroach',
            'couch',
            'crab',
            'crocodile',
            'cup',
            'dinosaur',
            'dolphin',
            'elephant',
            'flatfish',
            'forest',
            'fox',
            'girl',
            'hamster',
            'house',
            'kangaroo',
            'computer_keyboard',
            'lamp',
            'lawn_mower',
            'leopard',
            'lion',
            'lizard',
            'lobster',
            'man',
            'maple_tree',
            'motorcycle',
            'mountain',
            'mouse',
            'mushroom',
            'oak_tree',
            'orange',
            'orchid',
            'otter',
            'palm_tree',
            'pear',
            'pickup_truck',
            'pine_tree',
            'plain',
            'plate',
            'poppy',
            'porcupine',
            'possum',
            'rabbit',
            'raccoon',
            'ray',
            'road',
            'rocket',
            'rose',
            'sea',
            'seal',
            'shark',
            'shrew',
            'skunk',
            'skyscraper',
            'snail',
            'snake',
            'spider',
            'squirrel',
            'streetcar',
            'sunflower',
            'sweet_pepper',
            'table',
            'tank',
            'telephone',
            'television',
            'tiger',
            'tractor',
            'train',
            'trout',
            'tulip',
            'turtle',
            'wardrobe',
            'whale',
            'willow_tree',
            'wolf',
            'woman',
            'worm',
        ]

    }
    if dataset not in class_names.keys():
        raise Exception("Dataset class names not found")
    return class_names[dataset]


def permute_data(X, y):
    perm = np.arange(X.shape[0])
    np.random.shuffle(perm)
    X = X[perm]
    y = y[perm]
    return X, y


def augment_data(images, data_augmentation=False, is_training=False):
    """
    Performs random flips on input images
    :param images: Input images (tf placeholder)
    :param data_augmentation: bool flag for augmentation
    :return:
    """
    if data_augmentation:
        return tf.cond(is_training, lambda: tf.map_fn(lambda x: tf.image.random_flip_left_right(x),
                         images), lambda: images)
    else:
        return images


def save_image(image, image_size, save_dir, name=""):
    image += 1
    image *= 127.5
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = np.reshape(image, image_size)
    print(image.shape)
    misc.imsave(os.path.join(save_dir, "pred_image_%s.pdf" % name), image)


# pick up the same number of examples and labels in each class randomly from test dataset
def get_examples(x_test, y_test, num_examples, num_classes):
    x_examples = np.zeros([(num_classes * num_examples), (x_test.shape[1]), (x_test.shape[2]), (x_test.shape[3])])
    y_examples = np.zeros((num_classes * num_examples, num_classes))

    example_index = 0
    selected_index = np.zeros(x_test.shape[0], dtype='bool')
    countup_array = np.zeros(num_classes)  # the count of selected image in each class
    y_test_scalar = np.argmax(y_test, axis=1)
    for i in range(x_test.shape[0]):
        index = np.random.randint(0, x_test.shape[0])
        label = y_test_scalar[index,]

        if countup_array[label] < num_examples and selected_index[index] == False:
            x_examples[example_index, :, :, :] = x_test[index, :, :, :]
            y_examples[example_index, :] = y_test[index, :]
            countup_array[label] += 1
            example_index += 1
            selected_index[index] = True

    return x_examples, y_examples


# %% Tensorflow op utilities

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


def get_variable(weights, name, trainable=True):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init, shape=weights.shape, trainable=trainable)
    return var


def weight_variable(shape, stddev=0.02, name=None, trainable=True):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, shape=shape, initializer=tf.initializers.glorot_uniform(), trainable=trainable)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, shape=shape, initializer=tf.initializers.glorot_uniform())


def conv2d_basic(x, W, b=0):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def conv2d_basic_no_bias(x, W):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return conv


def conv2d_strided(x, W, b=0, stride=2):
    conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def conv2d_transpose_strided(x, W, b=0, output_shape=None, stride=2):
    # print x.get_shape()
    # print W.get_shape()
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x, padding='SAME'):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)


def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.losses.add_loss(tf.nn.l2_loss(var))


def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)


def get_tensor_size(tensor):
    """
    Return tensor size. Assumes first dimension is for batch size and is ignored.
    :param tensor: Input tensor
    :return: size of each batch element
    """
    size = 1
    for d in tensor.get_shape()[1:]:
        size = size * d.value
    return size


def train(loss_val, var_list, learning_rate, use_Adam=True, beta1=0.9, beta2=0.999, momentum=0.9, global_step=None):
    if use_Adam:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum, use_nesterov=True)

    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    # for grad, var in grads:
    #     add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads, global_step=global_step)


def model_accuracy(pred, labels):
    """
    Tensorflow op to return accuracy of predictions
    :param pred: tensorflow array containing predictions, shape = [None, n_classes]
    :param labels: tensorflow array of one hot labels, shape = [None, n_classes]
    :return: accuracy calculation op in tensorflow
    """
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    return tf.cast(correct_prediction, "float")


# %% Tensorflow utils copied for RESNET model


def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                           initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                            initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, 0.001)

    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    W = weight_variable(name='conv', shape=filter_shape)

    # conv_layer = conv2d_basic(input_layer, W)
    conv_layer = conv2d_basic_no_bias(input_layer, W)
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    W = weight_variable(name='conv', shape=filter_shape)
    # conv_layer = conv2d_basic(relu_layer, W)
    conv_layer = conv2d_basic_no_bias(relu_layer, W)
    return conv_layer


def residual_block(input_layer, output_channel, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            W = weight_variable(name='conv', shape=[3, 3, input_channel, output_channel])
            # conv1 = conv2d_basic(input_layer, W)
            conv1 = conv2d_basic_no_bias(input_layer, W)
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        # pooled_input = avg_pool_2x2(input_layer, padding='VALID')
        # padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
        #                                                               input_channel // 2]])
        padded_input = tf.pad(input_layer, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


# %% Model architectures

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )
    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w", trainable=False)
            bias = get_variable(bias.reshape(-1), name=name + "_b", trainable=False)
            current = conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = max_pool_2x2(current)
        net[name] = current
    return net


def vgg_network_architecture(input_data, scope_name="network", scope_reuse=False, logs_dir=""):
    print("setting up vgg initialized conv layers ...")
    model_data = get_model_data(logs_dir)
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])
    processed_image = tf.cast(input_data - mean_pixel, dtype=tf.float32)
    with tf.variable_scope(scope_name):
        image_net = vgg_net(weights, processed_image)
        # W = weight_variable([3, 3, 512, FLAGS.layer_size], name="W_conv0")
        # b = bias_variable([FLAGS.layer_size], name="b_conv0")
        # image_net[FLAGS.n_layers] = tf.nn.relu(conv2d_basic(image_net['relu4_1'], W, b))
    return image_net


def resnet_architecture(input_tensor_batch, n, scope_reuse=False):
    """
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param scope_reuse: obsolete parameter so far
    :return: last layer in the network. Not softmax-ed
    """
    layer_dict = {}
    with tf.variable_scope('conv0', reuse=scope_reuse):
        layer_dict[0] = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
    offset = 1
    for i in range(n):
        with tf.variable_scope('conv1_%d' % i, reuse=scope_reuse):
            if i == 0:
                layer_dict[offset] = residual_block(layer_dict[i + offset - 1], 16, first_block=True)
            else:
                layer_dict[i + offset] = residual_block(layer_dict[i + offset - 1], 16)
    offset = offset + n
    for i in range(n):
        with tf.variable_scope('conv2_%d' % i, reuse=scope_reuse):
            layer_dict[i + offset] = residual_block(layer_dict[i + offset - 1], 32)
    offset = offset + n
    for i in range(n):
        with tf.variable_scope('conv3_%d' % i, reuse=scope_reuse):
            layer_dict[i + offset] = residual_block(layer_dict[i + offset - 1], 64)
        # assert conv3.get_shape().as_list()[1:] == [8, 8, 64]
    offset = offset + n - 1
    with tf.variable_scope('fc', reuse=scope_reuse):
        in_channel = layer_dict[offset].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layer_dict[offset], in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        layer_dict[offset + 1] = tf.reduce_mean(relu_layer, [1, 2])

        assert layer_dict[offset + 1].get_shape().as_list()[-1:] == [64]

    return layer_dict, offset + 1
