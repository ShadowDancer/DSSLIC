import argparse
from datetime import datetime

from data.datasets import load_dataset
from model.DSSLIC import DSSLICModel
from utils.config import load_config


def run_main_gan(args, config):
    current_date = datetime.now()
    current_date = current_date.strftime('%Y-%m-%d_%H-%M-%S')
    experiment_name = current_date + '_' + config.dataset.name

    model_dir = '{}/{}'.format(args.checkpoints, experiment_name)

    batch_size = config.train.batch_size

    def input_fn():
        return load_dataset(config).batch(batch_size)  # dataset load ops must be called inside input_fn

    import tensorflow as tf
    run_config = tf.estimator.RunConfig().replace(
        log_step_count_steps=1,
        save_summary_steps=config.train.summary_iter,
        save_checkpoints_steps=config.train.checkpoint_iter,
        keep_checkpoint_max=config.train.checkpoint_keep_max
    )

    model = DSSLICModel(model_dir=model_dir, params=config, config=run_config)
    model.train(input_fn=input_fn, steps=config.train.steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DSSLIC')
    parser.add_argument('--config', help='Path to yml file file', default='config.yml')
    parser.add_argument('--name', help='Experiment name', default='config.yml')
    parser.add_argument('--checkpoints', default='./checkpoints', help='Directory with checkpoints')
    parser.add_argument('--restore-model', default=False, help='Path to restore model')

    args = parser.parse_args()
    config = load_config(args.config)

    import logging
    logging.getLogger().setLevel(logging.INFO)

    run_main_gan(args, config)
