import os

import cv2
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tqdm import tqdm

from face.dataset import AGFW
from utils.dataset.tf import FaceDataset, DataLoader


@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))


def GradCam(root, meta_path):
    ckpt_file = os.path.splitext(meta_path)[0]

    agfw_ds = FaceDataset(AGFW(
        root=root,
        attrib_name='gender',
        image_size=(160, 160),
        mean=(127.5, 127.5, 127.5),
        std=(128.0, 128.0, 128.0),
        multi_scale=False
    ), shuffle=False).create_dataset()

    eval_loader = DataLoader(agfw_ds, batch_size=32)

    conv_value = np.memmap('conv_value.npy', dtype=np.float16, mode='w+', shape=(len(eval_loader), 8, 8, 896))
    grad_value = np.memmap('grad_value.npy', dtype=np.float16, mode='w+', shape=(len(eval_loader), 8, 8, 896))
    y_value = np.memmap('y_labels.npy', dtype=np.uint8, mode='w+', shape=(len(eval_loader)))

    begin, end = 0, 0
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
            target_conv_layer = tf.get_default_graph().get_tensor_by_name(
                "InceptionResnetV1/Repeat_1/block17_10/Relu:0")

            loss = tf.losses.softmax_cross_entropy(labels_ph, logits_ph)

            target_conv_layer_grad = tf.gradients(loss, target_conv_layer)[0]

            # Guided backpropagtion back to input layer
            gb_grad = tf.gradients(loss, images_ph)[0]

            for iter, (fnames, images, y_labels) in enumerate(tqdm(eval_loader)):
                prob = sess.run(pred_probs, feed_dict={images_ph: images, phase_train_ph: False})

                y_labels = y_labels.reshape(len(y_labels), 1)
                one_hot_vector = np.eye(2)[y_labels.reshape(-1).astype(int)]

                gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run(
                    [gb_grad, target_conv_layer, target_conv_layer_grad],
                    feed_dict={images_ph: images, labels_ph: one_hot_vector, phase_train_ph: False})

                end = begin + len(fnames)
                conv_value[begin:end] = target_conv_layer_value.astype(np.float16)
                grad_value[begin:end] = target_conv_layer_grad_value.astype(np.float16)
                y_value[begin:end] = y_labels.astype(np.uint8).ravel()
                begin = end

                # for i in range(len(fnames)):
                #     # VGG16 use BGR internally, so we manually change BGR to RGB
                #     gradBGR = gb_grad_value[i]
                #     gradRGB = np.dstack((
                #         gradBGR[:, :, 2],
                #         gradBGR[:, :, 1],
                #         gradBGR[:, :, 0],
                #     ))
                #     heatmap = visualize(images[i], target_conv_layer_value[i], target_conv_layer_grad_value[i], gradRGB,
                #                         False)
                #     heatmaps.append(heatmap)


def load():
    conv_value = np.memmap('conv_value.npy', dtype=np.float16, mode='r', shape=(36299, 8, 8, 896))
    grad_value = np.memmap('grad_value.npy', dtype=np.float16, mode='r', shape=(36299, 8, 8, 896))
    y_value = np.memmap('y_labels.npy', dtype=np.uint8, mode='r', shape=(36299))

    female_index = y_value == 1
    heatmaps = []
    for conv, grad in zip(conv_value[female_index], grad_value[female_index]):
        output = conv  # [7,7,512]
        grads_val = grad  # [7,7,512]

        weights = np.mean(grads_val, axis=(0, 1))  # alpha_k, [512]
        cam = np.zeros(output.shape[0: 2], dtype=np.float32)  # [7,7]

        # Taking a weighted average
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        # Passing through ReLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)  # scale 0 to 1.0
        cam = resize(cam, (160, 160), preserve_range=True)

        cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
        heatmaps.append(cam_heatmap)
    heatmap = np.average(heatmaps, 0)


if __name__ == '__main__':
    meta_path = '/home/eugene/git/gender-classifier/checkpoint/model.ckpt-3.meta'
    # GradCam('/home/eugene/_DATASETS/Face/AGFW_cropped/align', meta_path)
    load()
