import argparse
import sys
from os import path

import tensorflow as tf

from data.datasets import load_dataset
from model.DSSLIC import DSSLICModel
from utils.config import load_config
from utils.misc import infer_optional_directory


def predict(args):
    """Trains model with settings specified in config

    :param args: Run configuration (directory names etc.)
    :param config: Model configuration (hyperparameters, datasets etc.)
    :return:
    """

    #config_file = infer_optional_directory(args.config, './config')


    model_dir = args.model
    if not path.isdir(model_dir):  # if checkpoint path get directory
        model_dir = path.dirname(model_dir)

    config = load_config(path.join(model_dir, 'config.yml'))

    restore_path = infer_optional_directory(args.model, args.checkpoints)
    def input_fn():
        d = load_dataset(config, config.train.batch_size, epochs=1, shuffle=args.shuffle)
        d = d.take(args.samples)
        return d

    model = make_model(None, restore_path, config)
    images = model.predict(input_fn=input_fn, checkpoint_path=restore_path)


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

    model = DSSLICModel(model_dir=model_dir, params=config, config=run_config, warm_start_from=restore_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DSSLIC - generates images predicted by model')

    parser.add_argument('--dataset', help='Overrides config dataset', default=None)
    parser.add_argument('--config', help='Config to use instead of config in model directory', default=None)
    parser.add_argument('--checkpoints', default='./checkpoints', help='Directory with checkpoints for path inference')

    parser.add_argument('model', default=None, help='Path to model checkpoint')
    parser.add_argument('samples', default=100, help='Samples to draw from dataset', type=int)
    parser.add_argument('out', help='Directory where output files will be saved')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset before sampling')


    args = parser.parse_args()

    import logging

    logging.getLogger().setLevel(logging.INFO)

    predict(args)
