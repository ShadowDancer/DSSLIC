import os
from glob import glob
from os import path

import tensorflow as tf
from PIL import Image


def load_dataset(config, batch_size, epochs=1, shuffle=True) -> tf.data.Dataset:
    """
    :param config: Loaded configuratin
    :return: tf.data.Dataset
    """

    def get_dataset_config(config):
        dataset_name = config.dataset.name
        dataset_cfg = getattr(config.dataset, dataset_name)
        return dataset_name, dataset_cfg

    d_name, d_cfg = get_dataset_config(config)

    dataset = None
    if d_name == 'ADE20K':
        dataset = _load_ade20k(d_cfg, batch_size, epochs, shuffle, config.dataset.shuffle_seed)

    return dataset


def _load_image(img_path):
    """ Loads and decodes image """
    image = tf.image.decode_jpeg(tf.read_file(img_path))
    return image


def _load_segmentation(img_path):
    """ Loads and decodes image """
    image = tf.image.decode_jpeg(tf.read_file(img_path))
    return image


def _resize_image(image, scale_shape, method:tf.image.ResizeMethod= tf.image.ResizeMethod.BICUBIC):
    image = tf.image.resize_images(image, [scale_shape[0], scale_shape[1]],
                                   method=method)
    image.set_shape(scale_shape)
    return image


def _load_ade20k(dataset_cfg, batch_size, epochs, shuffle, shuffle_seed):
    """ Loads ADE20K dataset, with segmentations

    :param dataset_cfg:
    :param batch_size:
    :param epochs: how many epochs should dataset return
    :param shuffle:
    :param shuffle_seed:
    :return: Dataset with fields image, segmentation, segmentation_one_hot (B,W,H,segmentation_classes)
    """
    dataset_path: str = dataset_cfg.path
    if not dataset_path.endswith('/'):
        dataset_path = dataset_path + os.sep
    dataset_path = dataset_path + '**/'

    image_files = glob(dataset_path + 'ADE_train_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].jpg', recursive=True)

    def image_file_to_segmentation(segmentation_path):
        directory = os.path.dirname(os.path.abspath(segmentation_path))
        file, ext = os.path.splitext(segmentation_path)
        file = file + '_seg'
        segmentation_path = path.join(directory, file + '.png')
        return segmentation_path

    segmentation_files = (image_file_to_segmentation(o) for o in image_files)
    data = [(image, segmentation) for image, segmentation in zip(image_files, segmentation_files) if
            os.path.isfile(segmentation) and Image.open(image).mode == 'RGB']
    path_ds = tf.data.Dataset.from_tensor_slices(data)  # (image_path, segmentation_path)

    def load_images(paths):
        """ Loads image from disk """
        image = _load_image(paths[0])
        segmentation = _load_segmentation(paths[1])

        return {
            'image_file': paths[0], 'segmentation_file': paths[1],
            'image': image, 'segmentation': segmentation
        }

    def filter_images(data):
        image_s = tf.shape(data['image'])
        r = image_s.shape[0] >= 512 and image_s.shape[1] >= 512
        print(r, image_s, image_s.shape[0] >= 512)
        return r

    def map_to_images(data):
        image = data['image']
        image = _resize_image(image, dataset_cfg.img_shape)
        image = tf.cast(image, tf.float32)
        image = (image / (255 / 2)) - 1  # normalize image image

        segmentation = data['segmentation']
        segmentation = _resize_image(segmentation, dataset_cfg.img_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # decode segmentation from R, G channels, where label = (R/10) * 256 + G
        segmentation_labels = ((segmentation[:, :, 0] // 10) * 256) + segmentation[:, :, 1]

        def segmentation_to_one_hot(labels, classes):
            """
            :param labels: Tensor with encoded labels
            :param classes: Number of output features
            :return: Tensor (256x256
            """
            one_hots = []
            for i in range(classes):
                r = tf.cast(tf.equal(labels, i + 1), tf.float32)  # class numbers star from 1
                one_hots.append(r)
            one_hots = tf.stack(one_hots, axis=2)

            one_hots.set_shape((labels.shape[0], labels.shape[1], classes))
            one_hots = tf.cast(one_hots, tf.float32)
            return one_hots

        segmentation_one_hot = segmentation_to_one_hot(segmentation_labels, dataset_cfg.segmentation_channels)

        result = dict(data)
        result['segmentation_one_hot'] = segmentation_one_hot
        result['segmentation'] = segmentation
        result['image'] = image
        return result, 0

    if shuffle:
        path_ds = path_ds.shuffle(30000, seed=shuffle_seed)
    path_ds = path_ds.repeat(epochs)

    image_ds = path_ds.map(load_images)
    image_ds.filter(filter_images)
    image_ds = image_ds.map(map_to_images)
    image_ds = image_ds.apply(tf.data.experimental.ignore_errors())
    if batch_size is not None:
        image_ds = image_ds.batch(batch_size)
    return image_ds
