import tensorflow as tf
from tensorflow.estimator import ModeKeys

from model.networks import configure_completion_net, configure_fine_net, configure_multiscale_discriminator


def image_tensor_to_rgb(image, add=1, scale=2):
    return (image + add) * (255 // scale)
    pass


def _model_fn(features, labels, mode: ModeKeys, params):
    """

    :param features: input batch
    :param _: labels batch, unused
    :param mode: training or prediction
    :param params: hyperparameters
    :return:
    """
    predictions, loss, train_op, train_hooks = None, None, None, []

    input_image, segmentation_rgb = features["image"], features["segmentation"]
    segmentation_one_hot = features["segmentation_one_hot"]


    config = params

    with tf.variable_scope("generator"):
        if config.segmentation.add_for_completion:
            completion_in = tf.concat([input_image, segmentation_one_hot], 3)
        else:
            completion_in = input_image
        coarse = configure_completion_net(completion_in, config)

        image_shape = tf.shape(input_image)
        image_shape = [image_shape[1], image_shape[2]]
        coarse_upsampled = tf.image.resize_images(coarse, image_shape, method=tf.image.ResizeMethod.BILINEAR)
        if config.segmentation.add_for_fine:
            fine_in = tf.concat([coarse_upsampled, segmentation_one_hot], 3)
        else:
            fine_in = coarse_upsampled
        residual = configure_fine_net(fine_in, config)
    reconstruction_image = coarse_upsampled + residual

    def gan_loss(discriminator_output, is_real):
        sign = -1 if is_real else 1
        return sign * tf.reduce_mean(discriminator_output)

    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        fake_discriminator_in = tf.concat([completion_in, reconstruction_image], axis=3)
        real_discriminator_in = tf.concat([completion_in, input_image], axis=3)
        fake_discriminator_result, fake_discriminator_features = configure_multiscale_discriminator(
            fake_discriminator_in, config)
        real_discriminator_result, real_discriminator_features = configure_multiscale_discriminator(
            real_discriminator_in, config)

    with tf.variable_scope("losses"):
        loss_d_adv_fake = gan_loss(fake_discriminator_result, False)
        loss_d_adv_real = gan_loss(real_discriminator_result, True)

        loss_g_adv = -gan_loss(fake_discriminator_result, False)

        mssim_loss = 0

        loss_l1 = tf.reduce_mean(tf.abs(input_image - reconstruction_image))

        # TODO: feature matching loss
        loss_gan_feature_matching = 0

        # TODO: VGG feature matching loss
        loss_vgg_feature_matching = 0

        # TODO: Print losses

        loss_d = (loss_d_adv_real + loss_d_adv_fake) * 0.5
        loss_g = loss_g_adv + loss_gan_feature_matching * 10 + loss_vgg_feature_matching * 10 + mssim_loss + loss_l1 * 20
        loss_g = loss_l1 * 20

    tf.summary.scalar('loss/generator', loss_g)
    tf.summary.scalar('loss/discriminator', loss_d)

    tf.summary.scalar('discriminator/loss_real', loss_d_adv_real)
    tf.summary.scalar('discriminator/loss_reconstruction', loss_d_adv_fake)
    tf.summary.scalar('generator/loss_adversarial', loss_g_adv)
    tf.summary.scalar('generator/loss_l1', loss_l1 * 20)

    def mse(x, y):
        b = x - y
        return tf.reduce_mean(b * b)

    tf.summary.scalar('generator/loss_l2', mse(input_image, reconstruction_image))
    tf.summary.scalar('generator/loss_l2_coarse', mse(input_image, coarse_upsampled))

    t = image_tensor_to_rgb

    for ins in range(0, 34, 5):
        sseg = tf.expand_dims(tf.concat(tf.unstack(segmentation_one_hot[:, :, :, ins:ins + 5], axis=3), 2), -1)

        tf.summary.image('segmentation_' + str(ins), t(sseg, 0, 1))
    image_summary = tf.concat([t(input_image), tf.cast(segmentation_rgb, tf.float32), t(coarse_upsampled), t(residual),
                               t(reconstruction_image)], 2)
    tf.summary.image('reconstruction input, segmentation. croase, residual, reconstructed image', image_summary)
    tf.summary.image('coarse', t(coarse))

    # TODO: decaying learning rate
    generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    optimizer_d = tf.train.AdamOptimizer(learning_rate=config.train.lr)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=config.train.lr)
    minimize_d = optimizer_d.minimize(loss_d, var_list=discriminator_vars)
    minimize_g = optimizer_g.minimize(loss_g, var_list=generator_vars, global_step=tf.train.get_global_step())

    if mode == ModeKeys.EVAL or mode == ModeKeys.TRAIN:
        # train_op = tf.group(minimize_d, minimize_g)
        train_op = minimize_g
        loss = loss_g + loss_d

    if mode == ModeKeys.PREDICT:
        # predictions = TODO
        pass

    return tf.estimator.EstimatorSpec(  # Retrun an Estimatorspec
        mode=mode,  # The mode tell's the spec which ops to use
        predictions=predictions,  # The predictions (only called if mode==PREDICT)
        loss=loss,  # Only used if mode!=PREDICT
        train_op=train_op,  # Only used if mode==TRAIN
        training_hooks=train_hooks
    )


class DSSLICModel(tf.estimator.Estimator):
    def __init__(self, model_dir=None, config=None, params=None):
        super(DSSLICModel, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config, params=params)

    pass
