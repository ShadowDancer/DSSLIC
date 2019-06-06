import argparse

import tensorflow as tf

from data.datasets import load_dataset
from model.DSSLIC import DSSLICModel
from utils.config import load_config, CopyConfigHook
from utils.misc import infer_optional_directory, get_new_model_directory


def train(args):
    """Trains model with settings specified in config

    :param args: Run configuration (directory names etc.)
    :param config: Model configuration (hyperparameters, datasets etc.)
    :return:
    """

    config_file = infer_optional_directory(args.config, './config')
    config = load_config(config_file)

    warm_start_path = infer_optional_directory(args.warm_start, args.checkpoints)
    resume_path = infer_optional_directory(args.resume, args.checkpoints)

    if warm_start_path is not None and resume_path is not None:
        raise RuntimeError('When resuming there is automatic warmstart from resume dir, warm start should be empty')

    if resume_path is None:
        model_dir = get_new_model_directory(args.checkpoints, args.name, config.dataset.name)
    else:
        warm_start_path = resume_path
        model_dir = resume_path

    def input_fn():
        return load_dataset(config, config.train.batch_size, epochs=-1, shuffle=config.train.shuffle)
    model = make_model(model_dir, warm_start_path, config)
    hooks = [CopyConfigHook(config_file, model_dir)]

    model.train(input_fn=input_fn, steps=config.train.steps, hooks=hooks)


def make_model(model_dir, restore_path, config):
    """Configures model

    :param model_dir: Directory to save checkpoints in
    :param restore_path: Checkpoint to restore
    :param config: Model configuration
    :return: Configured model
    """
    run_config = tf.estimator.RunConfig().replace(
        log_step_count_steps=1,
        save_summary_steps=config.train.summary_iter,
        save_checkpoints_steps=config.train.checkpoint_iter,
        keep_checkpoint_max=config.train.checkpoint_keep_max
    )

    if restore_path is not None:
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=restore_path,
                                            vars_to_warm_start=['generator', 'discriminator', 'global_step', 'beta'])
    else:
        ws = None
    model = DSSLICModel(model_dir=model_dir, params=config, config=run_config, warm_start_from=ws)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DSSLIC')
    parser.add_argument('--config', help='Path to yml file file', default='config.yml')
    parser.add_argument('--name', help='Experiment name', default=None)
    parser.add_argument('--checkpoints', default='./checkpoints', help='Directory with checkpoints')
    parser.add_argument('--warm-start', default=None, help='Loads weights of given model at start')
    parser.add_argument('--resume', default=None, help='Continues training of given model (warm-start must be empty)')

    args = parser.parse_args()

    import logging

    logging.getLogger().setLevel(logging.INFO)

    train(args)
