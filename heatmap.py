import tensorflow as tf
import numpy as np
import os

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from data_utils import prep_image, visualize

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    dtype = op.inputs[0].dtype
    return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)


def GradCam(image_path, meta_path):

    img = prep_image(image_path, (160, 160, 3))
    input_img = (img - 127.5) / 128
    input_img = input_img.astype(np.float32)
    input_img = input_img[np.newaxis, ...]

    ckpt_file = os.path.splitext(meta_path)[0]

    trained_model_graph = tf.Graph()
    with trained_model_graph.as_default():
        with trained_model_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            sess = tf.Session()
            new_saver = tf.train.import_meta_graph(meta_path)
            new_saver.restore(sess, ckpt_file)

            images_ph = tf.get_default_graph().get_tensor_by_name("image_batch:0")
            labels_ph = tf.get_default_graph().get_tensor_by_name("labels:0")
            phase_train_ph = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            pred_probs = tf.get_default_graph().get_tensor_by_name("prediction:0")
            logits_ph = tf.get_default_graph().get_tensor_by_name("logits/BiasAdd:0")
            target_conv_layer = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Repeat_1/block17_10/Relu:0")

            loss = tf.losses.softmax_cross_entropy(labels_ph, logits_ph)

            target_conv_layer_grad = tf.gradients(loss, target_conv_layer)[0]

            # Guided backpropagtion back to input layer
            gb_grad = tf.gradients(loss, images_ph)[0]

            prob = sess.run(pred_probs, feed_dict={images_ph: input_img, phase_train_ph:False})

            gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run(
                [gb_grad, target_conv_layer, target_conv_layer_grad],
                feed_dict={images_ph: input_img, labels_ph: [[1, 0]], phase_train_ph:False})


            gradBGR = gb_grad_value[0]
            gradRGB = np.dstack((
                gradBGR[:, :, 2],
                gradBGR[:, :, 1],
                gradBGR[:, :, 0],
            ))
            print(prob)
            visualize(img, target_conv_layer_value[0], target_conv_layer_grad_value[0], gradRGB)



if __name__ == '__main__':
    image_path = '/home/eugene/_DATASETS/Face/AGFW_cropped/128/female/age_45_49/14870.jpg'
    meta_path = '/home/eugene/git/gender-classifier/checkpoint/model.ckpt-3.meta'

    GradCam(image_path, meta_path)