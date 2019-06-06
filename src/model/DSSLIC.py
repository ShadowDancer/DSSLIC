import tensorflow as tf
from tensorflow.estimator import ModeKeys

from model.losses import ssim_loss, l1_loss, l2_loss, gan_loss, gan_feature_matching
from model.networks import configure_completion_net, configure_fine_net, configure_multiscale_discriminator
from model.utils import activation_to_image


def _model_fn(features, labels, mode: ModeKeys, params):
    """

    :param features: input batch
    :param _: labels batch, unused
    :param mode: training or prediction
    :param params: hyperparameters
    :return:
    """
    predictions, loss, train_op, train_hooks = None, None, None, []

    input_image_file = features["image_file"]
    input_image = features["image"]  # batch of rgb input images in [-1, 1]
    segmentation_rgb = features["segmentation"]  # z
    # batch of rgb segmentation in range [0, 255]
    segmentation_one_hot = features["segmentation_one_hot"]  # batch of segmentation one-hot encoded

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
    if config.nn.common.clip_reconstruction:
        reconstruction_image = tf.clip_by_value(reconstruction_image, -1, 1)
    if config.nn.common.use_residual_as_reconstruction:
        reconstruction_image = residual

    with tf.variable_scope("discriminator", reuse=False):
        fake_discriminator_in = tf.concat([completion_in, reconstruction_image], axis=3)
        real_discriminator_in = tf.concat([completion_in, input_image], axis=3)
        fake_discriminator_result, fake_discriminator_features = configure_multiscale_discriminator(
            fake_discriminator_in, config)
    with tf.variable_scope("discriminator", reuse=True):
        real_discriminator_result, real_discriminator_features = configure_multiscale_discriminator(
            real_discriminator_in, config)

    with tf.variable_scope("g_loss"):
        g_loss = []
        loss_cfg = config.train.loss

        loss_g_adv = gan_loss(fake_discriminator_result, is_real=True)  # generator wants make fakes like real

        d_feat_loss = gan_feature_matching(real_discriminator_features, fake_discriminator_features)

        vgg_feat_loss = 0
        if loss_cfg.reconstruction.vgg_feature_matching != 0:
            from model.losses import vgg_feature_matching
            vgg_feat_loss = vgg_feature_matching(input_image, reconstruction_image)

        ssim = ssim_loss(input_image + 1, reconstruction_image + 1, 2)
        ssim_coarse = ssim_loss(input_image + 1, coarse_upsampled + 1, 2)

        p = "reconstruction/"
        g_loss.append([p + "ssim", ssim, loss_cfg.reconstruction.ssim])
        g_loss.append([p + "l1", l1_loss(input_image, reconstruction_image), loss_cfg.reconstruction.l1])
        g_loss.append([p + "l2", l2_loss(input_image, reconstruction_image), loss_cfg.reconstruction.l2])

        g_loss.append([p + "adversarial", loss_g_adv, loss_cfg.reconstruction.g_adv])
        g_loss.append([p + "discriminator_feature_matching", d_feat_loss, loss_cfg.reconstruction.d_feature_matching])
        g_loss.append([p + "vgg_feature_matching", vgg_feat_loss, loss_cfg.reconstruction.vgg_feature_matching])

        p = "coarse/"
        g_loss.append([p + "ssim", ssim_coarse, loss_cfg.coarse.ssim])
        g_loss.append([p + "l1", l1_loss(input_image, coarse_upsampled), loss_cfg.coarse.l1])
        g_loss.append([p + "l2", l2_loss(input_image, coarse_upsampled), loss_cfg.coarse.l2])

        loss_g = sum(l[1] * l[2] for l in g_loss)
        tf.summary.scalar('loss/reconstruction', loss_g)

        for name, loss, factor in g_loss:
            if factor != 0:
                tf.summary.scalar(name, loss * factor)

        # discriminator adversarial losses
        loss_d_adv_fake = gan_loss(fake_discriminator_result, False)
        loss_d_adv_real = gan_loss(real_discriminator_result, True)
        loss_d = (loss_d_adv_real + loss_d_adv_fake) * 0.5

    tf.summary.scalar("discriminator/real", loss_d_adv_real)
    tf.summary.scalar("discriminator/fake", loss_d_adv_fake)
    tf.summary.scalar('loss/discriminator', loss_d)
    # summary images
    t = activation_to_image
    seg = tf.cast(segmentation_rgb, tf.float32)
    image_summary = tf.concat([t(input_image), seg, t(coarse_upsampled), t(residual), t(reconstruction_image)], 2)
    image_summary = tf.cast(image_summary, tf.uint8)  # cast to uint8 to avoid scaling
    tf.summary.image('reconstruction input, segmentation. croase, residual, reconstructed image', image_summary)
    tf.summary.image('coarse', tf.cast(t(coarse), tf.uint8))

    # TODO: decaying learning rate
    generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    optimizer_d = tf.train.AdamOptimizer(learning_rate=config.train.lr)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=config.train.lr)

    minimize_g = optimizer_g.minimize(loss_g, var_list=generator_vars, global_step=tf.train.get_global_step())
    if loss_cfg.reconstruction.g_adv != 0:
        minimize_d = optimizer_d.minimize(loss_d, var_list=discriminator_vars)
        train_op = tf.group(minimize_d, minimize_g)
    else:
        train_op = tf.group(minimize_g)

    if mode == ModeKeys.EVAL or mode == ModeKeys.TRAIN:
        train_op = train_op
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
    def __init__(self, model_dir=None, config=None, params=None, warm_start_from=None):
        super(DSSLICModel, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config, params=params,
                                          warm_start_from=warm_start_from)

    pass
