import tensorflow as tf

from model.utils import activation_to_image
from model.utils import var_scope


def ssim_loss(rgb_a, rgb_b, dynamic_range):
    """Computes ssim of a batch
    :param rgb_a: batch of rgb tensors in range (0, range)
    :param rgb_b:
    :param dynamic_range: dynamic range of images (max pixel value)
    :return:
    """
    a = tf.image.rgb_to_grayscale(rgb_a)
    b = tf.image.rgb_to_grayscale(rgb_b)
    return -tf.reduce_mean(tf.image.ssim(a, b, dynamic_range))


def l1_loss(x, y):
    """Absoute error"""
    return tf.reduce_mean(tf.abs(x - y))


def l2_loss(x, y):
    """Mean squared error"""
    b = x - y
    return tf.reduce_mean(b * b)


def gan_loss(discriminator_output, is_real: bool):
    """GAN adversarial loss

    :param discriminator_output: output Tensor from multiscale discriminator
    :param is_real: is this output for real images, or generated
    :return:
    """
    loss = tf.zeros([])
    for d in range(len(discriminator_output)):
        out = discriminator_output[d]
        out_shape = tf.shape(out)
        target_label = tf.ones(out_shape) if is_real else tf.zeros(out_shape)
        loss += l2_loss(out, target_label)
    return loss


def gan_feature_matching(d_features_real, d_features_fake):
    """Compares distance between discriminator features on real and fake images

    :param d_features_real:
    :param d_features_fake:
    :return:
    """
    with tf.variable_scope("feature_matching"):
        d_num = len(d_features_real)
        assert len(d_features_real) == len(d_features_fake)

        feature_matching_loss = tf.zeros([])
        d_num_factor = 1 / d_num  # scale to number of discriminators
        for d in range(d_num):  # iterate discriminator results

            layers_real = d_features_real[d]
            layers_fake = d_features_fake[d]

            l_num_factor = 1 / len(layers_real)
            assert len(layers_real) == len(layers_fake)  # scale to number of layers
            for l in range(len(d_features_real)):  # iterate layers of discriminator
                layer_real = layers_real[l]
                layer_fake = layers_fake[l]
                feature_matching_loss += l1_loss(layer_real, layer_fake) * d_num_factor * l_num_factor
        return feature_matching_loss


@var_scope('vgg_feature_matching')
def vgg_feature_matching(images_a, images_b):
    """
    Returns sum of l1 distances between features in vgg19 network
    :param images_a: 
    :param images_b:
    :return:
    """
    import tensornets as tnets
    def preprocess(x):
        x = x[:, :, :, ::-1]

        channels = [x[:, :, :, 0] - 103.939,
                    x[:, :, :, 1] - 116.779,
                    x[:, :, :, 2] - 123.68]
        x = tf.stack(channels, axis=3)
        x = tf.image.resize_images(x, (224, 224))  # vgg expects images in shape 224, 224
        return x

    def get_vgg_features(input):
        """Gets activation of vgg network
        :param input: input tensor with image
        :return: activations of selected layers
        """
        input = preprocess(input)
        model = tnets.VGG19(input)
        model.pretrained()
        print(model.pretrained)

        feature_names = ['vgg19/conv3/4/Relu:0', 'vgg19/conv4/4/Relu:0', 'vgg19/conv5/4/Relu:0']

        middles = model.get_middles()
        result = []

        for tensor in middles:
            for name in feature_names:
                if tensor.name.endswith(name):
                    result.append(tensor)

        return result

    images_a = activation_to_image(images_a)
    images_b = activation_to_image(images_b)

    with tf.variable_scope("vgg", reuse=False):
        featues_a = get_vgg_features(images_a)
    with tf.variable_scope("vgg", reuse=True):
        features_b = get_vgg_features(images_b)

    loss = 0
    weights = [1.0 / 8, 1.0 / 4, 1.0]
    for i, (a_feature, b_feature) in enumerate(zip(featues_a, features_b)):
        loss += weights[i] * l1_loss(a_feature, b_feature)

    return loss
