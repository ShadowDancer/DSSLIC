import tensorflow as tf


def reflectionPadding2D(features, pad):
    """ Applies reflection padding over 4d features tensor

    :param features: 4d input tensor
    :param pad: widh, heigh
    :return: tensorf with edges padded with reflection padding
    """

    w_pad, h_pad = pad, pad
    return tf.pad(features, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT', name='reflectionPadding2D')
