import cv2
import numpy as np
import skimage
import skimage.io
import skimage.transform
from matplotlib import pyplot as plt
from skimage.transform import resize
from PIL import Image

def make_image_3_channels(im):
    if np.ndim(im) == 2:
        im = np.expand_dims(im, 2)
    if im.shape[2] == 1:
        im = np.repeat(np.expand_dims(im[:, :, 0], 2), 3, axis=2)
    if im.shape[2] > 3:
        im = im[:, :, 0:3]
    return im


def letterbox_image(image, target_size):
    '''resize image without changing aspect ratio, then paste it to a gray canvas with target_size '''
    image = Image.fromarray(image)
    iw, ih = image.size
    w, h, c = target_size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return np.array(new_image)


def prep_image(filename, dims):
    img = skimage.io.imread(filename)
    img = make_image_3_channels(img)
    img = letterbox_image(img, dims)
    return img

# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1


def visualize(image, conv_output, conv_grad, gb_viz):
    output = conv_output  # [7,7,512]
    grads_val = conv_grad  # [7,7,512]
    print("grads_val shape:", grads_val.shape)
    print("gb_viz shape:", gb_viz.shape)

    weights = np.mean(grads_val, axis=(0, 1))  # alpha_k, [512]
    cam = np.zeros(output.shape[0: 2], dtype=np.float32)  # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)  # scale 0 to 1.0
    cam = resize(cam, (160, 160), preserve_range=True)

    # img = image.astype(float)
    # img -= np.min(img)
    # img /= img.max()
    # print(img)
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    # cam = np.float32(cam) + np.float32(img)
    # cam = 255 * cam / np.max(cam)
    # cam = np.uint8(cam)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(image)
    ax.set_title('Input Image')

    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(131)
    imgplot = plt.imshow(cam_heatmap)
    ax.set_title('Grad-CAM')

    gb_viz = np.dstack((
        gb_viz[:, :, 0],
        gb_viz[:, :, 1],
        gb_viz[:, :, 2],
    ))
    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()

    ax = fig.add_subplot(132)
    imgplot = plt.imshow(gb_viz)
    ax.set_title('guided backpropagation')

    gd_gb = np.dstack((
        gb_viz[:, :, 0] * cam,
        gb_viz[:, :, 1] * cam,
        gb_viz[:, :, 2] * cam,
    ))
    ax = fig.add_subplot(133)
    imgplot = plt.imshow(gd_gb)
    ax.set_title('guided Grad-CAM')

    plt.show()