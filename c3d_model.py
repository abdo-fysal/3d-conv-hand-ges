"""Builds the C3D network.

Implements the inference pattern for model building.
inference_c3d(): Builds the model as far as is required for running the network
forward to make predictions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
import numpy as np

################RECURRENT_LAYER#####################################################33



#with tf.variable_scope('local8') as scope:
  #c = _variable_on_cpu('c', [state_dim, 1], tf.zeros_initializer())

    # c = tf.Variable(tf.zeros(shape=(state_dim, 1)))
#with tf.variable_scope('local9') as scope:
  #s = _variable_on_cpu('s', [state_dim, 1], tf.zeros_initializer())
    # s = tf.Variable(tf.zeros(shape=(state_dim, 1)))

#with tf.variable_scope('local10') as scope:
  #U = _variable_on_cpu('U', [4, state_dim, inp_dim],
                    #    tf.random_uniform(shape=(4, state_dim, inp_dim), minval=-np.sqrt(1. / inp_dim),
                                   #       maxval=np.sqrt(1. / inp_dim)))

    # U = tf.Variable(tf.random_uniform(shape=(4, state_dim, inp_dim),minval=-np.sqrt(1. / inp_dim),maxval=np.sqrt(1. / inp_dim)))

#with tf.variable_scope('local11') as scope:
  #W = _variable_on_cpu('W', [4, state_dim, state_dim],
                        #tf.random_uniform(shape=(4, state_dim, state_dim), minval=-np.sqrt(1. / state_dim),
                                         # maxval=np.sqrt(1. / state_dim)))

#with tf.variable_scope('local12') as scope:
#  b = _variable_on_cpu('b', [4, state_dim, 1], tf.ones_initializer())

    # b = tf.Variable(tf.ones(shape=(4, state_dim, 1)))
#with tf.variable_scope('local13') as scope:
  #V = _variable_on_cpu('V', [out_dim, state_dim],
                     #   tf.random_uniform(shape=(out_dim, state_dim), minval=-np.sqrt(1. / state_dim),
                      #                    maxval=np.sqrt(1. / state_dim)))

    # V = tf.Variable(tf.random_uniform(shape=(out_dim, state_dim),minval=-np.sqrt(1. / state_dim),maxval=np.sqrt(1. / state_dim)))
#with tf.variable_scope('local14') as scope:
  #d = _variable_on_cpu('d', [out_dim, 1], tf.ones_initializer())

    # d = tf.Variable(tf.ones(shape=(out_dim, 1)))

#learn_r = tf.placeholder(tf.float32)
#decay_r = tf.placeholder(tf.float32)

  # Define the variable to hold the adaptive learning rates
#with tf.variable_scope('local15') as scope:
  #mU = _variable_on_cpu('mU', [4, state_dim, inp_dim], tf.zeros_initializer())
    # self.mU = tf.Variable(tf.zeros(shape=self.U.shape))

#with tf.variable_scope('local16') as scope:
 # mW = _variable_on_cpu('mW', [4, state_dim, state_dim], tf.zeros_initializer())
    # self.mW = tf.Variable(tf.zeros(shape=self.W.shape))
#with tf.variable_scope('local17') as scope:
  #mb = _variable_on_cpu('mb', [4, state_dim, 1], tf.zeros_initializer())
    # self.mb = tf.Variable(tf.zeros(shape=self.b.shape))
#with tf.variable_scope('local18') as scope:
  #mV = _variable_on_cpu('mV', [out_dim, state_dim], tf.zeros_initializer())
    # self.mV = tf.Variable(tf.zeros(shape=self.V.shape))

#with tf.variable_scope('local19') as scope:
  #md = _variable_on_cpu('md', [out_dim, 1], tf.zeros_initializer())
    # self.md = tf.Variable(tf.zeros(shape=self.d.shape))

#def forward_step(acc, word):
  #c, s, output = acc

    # LSTM layer
  #i = tf.sigmoid(tf.reshape(U[0, :, word], (-1, 1)) + tf.matmul(W[0], s) + b[0])
  #f = tf.sigmoid(tf.reshape(U[1, :, word], (-1, 1)) + tf.matmul(W[1], s) + b[1])
  #o = tf.sigmoid(tf.reshape(U[2, :, word], (-1, 1)) + tf.matmul(W[2], s) + b[2])
  #g = tf.tanh(tf.reshape(U[3, :, word], (-1, 1)) + tf.matmul(W[3], s) + b[3])

  #c = f * c + g * i
  #s = tf.tanh(c) * o

    # Output calculation
  #output = tf.matmul(V, s) + d
  #output = tf.minimum(tf.maximum(0, output), 4)

  #return [c, s, output]

    #########################################################################################

#ce_init = [c, s, tf.zeros(shape=(out_dim, 1))]
 # results = tf.scan(forward_step, local7, ce_init)
 # softmax_linear = results[2]












###########################################################################################3333333






# The number of classes of the dataset
NUM_CLASSES = 7

# Images are cropped to (CROP_SIZE, CROP_SIZE)
CROP_SIZE = 128
CHANNELS = 3
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 18750 
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 56321
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 6250
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 18773

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 16.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001       # Initial learning rate.

# Number of frames per video clip
NUM_FRAMES_PER_CLIP = 16

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # TODO: update this function according to the video input. previously is image input

  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, wd=None):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with "Xavier" initialization.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(
      name,
      shape,
      tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def conv_3d(kernel_name, biases_name, input, kernel_shape, 
            biases_shape, kernel_wd, biases_wd=None):
  kernel = _variable_with_weight_decay(kernel_name, kernel_shape, kernel_wd)
  conv = tf.nn.conv3d(input, kernel, [1, 1, 1, 1, 1], padding='SAME')
  biases = _variable_with_weight_decay(biases_name, biases_shape, biases_wd)
  pre_activation = tf.nn.bias_add(conv, biases)
  return pre_activation


def max_pool(name, l_input, k):
  return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1],
                          strides=[1, k, 2, 2, 1], padding='SAME',
                          name=name)


def inference_c3d(videos, _dropout=1, features=False):
  """Generate the 3d convolution classification output according to the input
    videos

  Args:
    videos: Data Input, the shape of the Data Input is 
      [batch_size, sequence_size, height, weight, channel]
  Return:
    out: classification result, the shape is [batch_size, num_classes]
  """
  # Conv1 Layer
  with tf.variable_scope('conv1') as scope:
    conv1 = conv_3d('weight', 'biases', videos, 
                    [3, 3, 3, CHANNELS, 64], [64], 0.0005)
    conv1 = tf.nn.relu(conv1, name=scope.name)
    _activation_summary(conv1)

  # pool1 
  pool1 = max_pool('pool1', conv1, k=1)

  # Conv2 Layer
  with tf.variable_scope('conv2') as scope:
    conv2 = conv_3d('weight', 'biases', pool1,
                    [3, 3, 3, 64, 128], [128], 0.0005)
    conv2 = tf.nn.relu(conv2, name=scope.name)
    _activation_summary(conv2)

  # pool2
  pool2 = max_pool('pool2', conv2, k=2)

  # Conv3 Layer
  with tf.variable_scope('conv3') as scope:
    conv3 = conv_3d('weight_a', 'biases_a', pool2,
                    [3, 3, 3, 128, 256], [256], 0.0005)
    conv3 = tf.nn.relu(conv3, name=scope.name+'a')
    conv3 = conv_3d('weight_b', 'biases_b', conv3,
                    [3, 3, 3, 256, 256], [256], 0.0005)
    conv3 = tf.nn.relu(conv3, name=scope.name+'b')
    _activation_summary(conv3)

  # pool3
  pool3 = max_pool('pool3', conv3, k=2)

  # Conv4 Layer
  with tf.variable_scope('conv4') as scope:
    conv4 = conv_3d('weight_a', 'biases_a', pool3,
                    [3, 3, 3, 256, 512], [512], 0.0005)
    conv4 = tf.nn.relu(conv4, name=scope.name+'a')
    conv4 = conv_3d('weight_b', 'biases_b', conv4,
                    [3, 3, 3, 512, 512], [512], 0.0005)
    conv4 = tf.nn.relu(conv4, name=scope.name+'b')
    _activation_summary(conv4)

  # pool4
  pool4 = max_pool('pool4', conv4, k=2)

  # Conv5 Layer
  with tf.variable_scope('conv5') as scope:
    conv5 = conv_3d('weight_a', 'biases_a', pool4,
                    [3, 3, 3, 512, 512], [512], 0.0005)
    conv5 = tf.nn.relu(conv5, name=scope.name+'a')
    conv5 = conv_3d('weight_b', 'biases_b', conv5,
                    [3, 3, 3, 512, 512], [512], 0.0005)
    conv5 = tf.nn.relu(conv5, name=scope.name+'b')
    _activation_summary(conv5)

  # pool5
  pool5 = max_pool('pool5', conv5, k=2)#conv5

  # local6
  with tf.variable_scope('local6') as scope:
    weights = _variable_with_weight_decay('weights', [8192, 4096], 0.0005)
    biases = _variable_with_weight_decay('biases', [4096])
    pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
    local6 = tf.reshape(pool5, [-1, weights.get_shape().as_list()[0]])
    local6 = tf.nn.relu(tf.matmul(local6, weights) + biases, name=scope.name)
    if features:
      return local6
    local6 = tf.nn.dropout(local6, _dropout)
    _activation_summary(local6)

  # local7
  with tf.variable_scope('local7') as scope: 
    weights = _variable_with_weight_decay('weights', [4096, 4096], 0.0005)
    biases = _variable_with_weight_decay('biases', [4096])
    local7 = tf.nn.relu(tf.matmul(local6, weights) + biases, name=scope.name)
    local7 = tf.nn.dropout(local7, _dropout)
    _activation_summary(local7)

  with tf.variable_scope('softmax_lineaerr') as s:
    num_inputs = 4096

    num_neurons = 500

    num_outputs = 7
    ix=tf.reshape(local7,[tf.shape(local7)[0],1,tf.shape(local7)[1]])
    ix.set_shape([None,1,4096])




    cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs)

    softmax_linear, states = tf.nn.dynamic_rnn(cell, ix, dtype=tf.float32)
    print(softmax_linear.shape)
    softmax_linear=tf.reshape(softmax_linear,[tf.shape(softmax_linear)[0],tf.shape(softmax_linear)[2]])
    softmax_linear.set_shape([None, 7])
  #_activation_summary(softmax_linear)







  #linear_layer(Wx + b)
  #with tf.variable_scope('softmax_lineaer') as scope:
    #weights = _variable_with_weight_decay('weights', [4096, NUM_CLASSES], 0.0005)
    #biases = _variable_with_weight_decay('biases', [NUM_CLASSES])
    #softmax_linear = tf.add(tf.matmul(local7, weights), biases, name=scope.name)
    #_activation_summary(softmax_linear)

  return softmax_linear


def loss(logits, labels):
  """Add L2Loss to all the trainable variable

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference()
    labels: Labels from dataset. 1-D tensor of shape [batch_size]

  Returns:
    Loss tensor of type float
  """
  # Calculate the average cross entropy loss across the batch
  zero = tf.constant(0, dtype=tf.int32)
  labels=tf.cast(labels, tf.int32)
  where = tf.not_equal(labels, zero)
  indices = tf.where(where)
  values = tf.gather_nd(labels, indices)
  sparse = tf.SparseTensor(indices, values, (10,7))
  #logits = tf.reshape(logits, [10, -1, 7])

  # Time major
  #logits = tf.transpose(logits, (1, 0, 2))

 # loss = tf.nn.ctc_loss(inputs=logits,labels=sparse,sequence_length= [10])
 #   cost = tf.reduce_mean(loss)







  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the across entropy loss plus all the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')



