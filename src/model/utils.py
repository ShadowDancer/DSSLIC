import tensorflow as tf


def activation_to_image(image):
    """ Transforms image from [-1, 1] range to [0, 255]

    :param image: Tensor with image in range [-1, 1] (output of tanh activation)
    :return: Image in range [0,255]
    """
    return (image + 1) * (255 / 2)
    pass


def var_scope(name):
    """Wraps function call in variable scope"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            scope = kwargs.get('scope', name)
            reuse = kwargs.get('reuse', None)
            with tf.variable_scope(scope, reuse=reuse):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def gradient_summary(grads_and_vars):
    for g, v in grads_and_vars:
        name: str = v.name.replace(":", "_")
        tf.summary.histogram(name, v)
        tf.summary.histogram(name + '_grad', g)
