import argparse
import os
from datetime import datetime
from os import path

import numpy as np
import tensorflow as tf
from PIL import Image

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

    model_dir = args.model
    if not path.isdir(model_dir):  # if checkpoint path get directory
        model_dir = path.dirname(model_dir)

    config = load_config(path.join(model_dir, 'config.yml'))

    restore_path = infer_optional_directory(args.model, args.checkpoints)

    def input_fn():
        d = load_dataset(config, 1, epochs=1, shuffle=args.shuffle)
        d = d.take(args.samples)
        return d

    out_dir = args.out

    if out_dir is None:
        current_date = datetime.now()
        current_date = current_date.strftime('%Y-%m-%d__%H-%M-%S')

        out_dir = 'results-' + current_date

    os.makedirs(out_dir, exist_ok=False)

    model = make_model(None, restore_path, config)
    predictions = model.predict(input_fn=input_fn, checkpoint_path=restore_path)

    for i, prediction in enumerate(predictions):
        input_image_np = prediction["input_image"]
        segmentation_labels_np = prediction["segmentation"]
        coarse_np = prediction["coarse_image"]
        residual_np = prediction["residual_image"]
        reconstruction_image_np = prediction["reconstruction_image"]
        input_image_file = prediction["input_file"].decode('ASCII')

        base_file_name = os.path.splitext(os.path.basename(input_image_file))[0]

        input_image = convert_image(input_image_np)
        input_image.save(path.join(out_dir, base_file_name + '.bmp'), "BMP")

        reconstruction_image = convert_image(reconstruction_image_np)
        reconstruction_image.save(path.join(out_dir, base_file_name + '_reconstruction.bmp'), "BMP")

        coarse_image = convert_image(coarse_np)
        coarse_image.save(path.join(out_dir, base_file_name + '_coarse.bmp'), "BMP")

        segmentation = Image.fromarray(np.squeeze(segmentation_labels_np), "L")
        segmentation.save(path.join(out_dir, base_file_name + '_segmentation.bmp'), "BMP")

        r_min, r_max, residual_scaled = normalize_resitual(residual_np)
        residual_image = Image.fromarray(residual_scaled, "RGB")
        residual_image.save(path.join(out_dir, base_file_name + '_residual.bmp'), "BMP")

        np.savetxt(path.join(out_dir, base_file_name + 'residual_scale.txt'), np.array([r_min, r_max]))
        print('Processing', i, '/', args.samples)


def normalize_resitual(residual):
    """
    Scales residaul to [0-255]

    :param residual: numpy array
    :return: r_min, r_max, scaled, such that residual = scaled / r_max + r_min
    """
    r_min = np.min(residual)
    residual = residual - r_min
    r_max = 255 / np.max(residual)
    residual = residual * r_max
    return r_min, r_max, residual


def convert_image(numpy_image):
    """
    Converts [-1,1] numpy array to [0, 255] PIL image

    :param numpy_image:
    :return:
    """
    scaled = (numpy_image + 1) * (255 / 2)
    scaled = scaled.astype(np.uint8)
    return Image.fromarray(scaled)


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
    parser.add_argument('--out', help='Directory where output files will be saved', default=None)
    parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset before sampling')

    args = parser.parse_args()

    import logging

    logging.getLogger().setLevel(logging.INFO)

    predict(args)
