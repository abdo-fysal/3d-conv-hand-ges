from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import math
import numpy as np
import input_data
import c3d_model
import cv2


import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import PIL.Image as Image
import random
import time
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('gpu_num', 1,
                            """How many GPUs to use""")
tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'result',
                            """Check point directory.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('num_examples', 5000,
                            """Number of examples to run.""")


def data_input(tmp_data,num_frames_per_clip=16, crop_size=128, shuffle=False):


    data = []

    img_datas = []
    if(len(tmp_data)!=0):
      for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if(img.width>img.height):
          scale = float(crop_size)/float(img.height)
          img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
          scale = float(crop_size)/float(img.width)
          img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
        img = img[int((img.shape[0] - crop_size)/2):int((img.shape[0] - crop_size)/2) + crop_size, int((img.shape[1] - crop_size)/2):int((img.shape[1] - crop_size)/2) + crop_size,:]
        img_datas.append(img)
      data.append(img_datas)


    np_arr_data = np.array(data).astype(np.float32)

    return np_arr_data
def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder


def evaluate(d):
  with tf.Graph().as_default() as g:
    # Get the image and the labels placeholder
    images_placeholder, labels_placeholder = placeholder_inputs(1)

    # Build the Graph that computes the logits predictions from the inference
    # model.
    with tf.variable_scope('c3d_var'):

        logits = c3d_model.inference_c3d(images_placeholder)




    #top_k_op = tf.nn.in_top_k(logits, labels_placeholder, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        c3d_model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            while True :

                if len(d)==16:

                    in_x = data_input(d)
                    d=[]


        #cv2.destroyAllWindows()


                    l=sess.run([logits],feed_dict={images_placeholder:in_x})
                    l=np.array(l)
                    l = np.reshape(l, (np.product(l.shape),))
                    print(l.shape)
                    p=sess.run(tf.nn.softmax(l))
                    pred=sess.run(tf.argmax(p))
                    print(pred)
        except Exception as e:

            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


        cv2.destroyAllWindows()


def main(_):
  evaluate()


if __name__ == '__main__':
  tf.app.run()



