import os
from glob import glob
from os import path

import tensorflow as tf
from PIL import Image


def load_dataset(config) -> tf.data.Dataset:
    """
    :param config: Loaded configuratin
    :return: tf.data.Dataset
    """

    def get_dataset_config(config):
        dataset_name = config.dataset.name
        dataset_cfg = getattr(config.dataset, dataset_name)
        return dataset_name, dataset_cfg

    name, dataset_cfg = get_dataset_config(config)

    if name == 'ADE20K':
        return _load_ade20k(dataset_cfg)


def _load_image(img_path, scale_shape):
    """ Loads and decodes image """
    image = tf.image.decode_jpeg(tf.read_file(img_path))
    image = tf.image.resize_images(image, [scale_shape[0], scale_shape[1]])
    image.set_shape(scale_shape)  # 3 channels
    image = tf.cast(image, tf.float32)
    image = (image / (255 / 2)) - 1  # normalize image image
    return image


def _load_segmentation(img_path, scale_shape):
    """ Loads and decodes image """
    image = tf.image.decode_png(tf.read_file(img_path), channels=scale_shape[2])
    image = tf.image.resize_images(image, [scale_shape[0], scale_shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image.set_shape(scale_shape)
    return image


def _load_ade20k(dataset_cfg):
    dataset_path: str = dataset_cfg.path
    if not dataset_path.endswith('/'):
        dataset_path = dataset_path + os.sep
    dataset_path = dataset_path + '**/'

    image_files = glob(dataset_path + 'ADE_train_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].jpg', recursive=True)

    def orig_to_fake_path(orig_path):
        dir = os.path.dirname(os.path.abspath(orig_path))
        file, ext = os.path.splitext(orig_path)
        file = file + '_seg'
        fake_path = path.join(dir, file + '.png')
        file = file + '_one_hot'
        fake_path2 = path.join(dir, file + '.png')
        return fake_path, fake_path2

    segmentation_files = (orig_to_fake_path(o) for o in image_files)
    data = [(orig, fake, fake2) for orig, (fake, fake2) in zip(image_files, segmentation_files) if
            os.path.isfile(fake) and Image.open(orig).mode == 'RGB']
    path_ds = tf.data.Dataset.from_tensor_slices(data)  # (image_path, segmentation_path)

    def map_to_images(paths):
        image = _load_image(paths[0], dataset_cfg.img_shape)
        # load rgb segmentation for display purposes?
        segmentation = _load_segmentation(paths[1], dataset_cfg.img_shape)
        # load processed grayscale image, where each pixel is class id (1, 2, 3) etc.
        label_s = list(dataset_cfg.img_shape)
        label_s[2] = 1
        segmentation_labels = _load_segmentation(paths[2], label_s)

        def segmentation_to_one_hot(labels, classes):
            """
            :param labels: Tensor with encoded labels
            :param classes: Number of output features
            :return: Tensor (256x256
            """
            result = []
            for i in range(classes):
                r = tf.cast(tf.equal(labels, i + 1), tf.float32)  # class numbers star from 1
                result.append(r[:, :, 0])
            result = tf.stack(result, axis=2)

            result.set_shape((labels.shape[0], labels.shape[1], classes))
            result = tf.cast(result, tf.float32)
            return result

        segmentation_one_hot = segmentation_to_one_hot(segmentation_labels, dataset_cfg.segmentation_channels)
        return {'image': image, 'segmentation': segmentation, 'segmentation_one_hot': segmentation_one_hot}, 0

    image_ds = path_ds.map(map_to_images)
    image_ds = image_ds.apply(tf.data.experimental.ignore_errors())
    return image_ds
