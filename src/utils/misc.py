from os import path
from datetime import datetime


def infer_optional_directory(target_path, optional_directory):
    """ If path does not exist checks optional directory for that path (file or dir)

    :param target_path: path, name of checkpoint directory or path to model to restore
    :param optional_directory: directory to check if given path is incorrect
    :return: path to file if found, or unaltered target_path otherwise
    """

    if target_path is None:
        return None
    restore_path = target_path
    if restore_path is not None:
        if not path.exists(restore_path):
            checkpoint_relative_path = path.join(optional_directory, restore_path)
            if not path.isabs(restore_path) and path.exists(checkpoint_relative_path):
                restore_path = checkpoint_relative_path  # restoring

    return restore_path


def get_new_model_directory(checkpoint_dir, prefix, dataset_name):
    """Returns path to new directory in checkpoints_dir

    :param checkpoint_dir: Root directory
    :param prefix: Prefix to add to directory (name of experiment)
    :param dataset_name: Name of dataset to append to dir name
    :return:
    """
    current_date = datetime.now()
    current_date = current_date.strftime('%Y-%m-%d__%H-%M-%S')
    directory_name = current_date + '__' + dataset_name

    if prefix is not None:
        directory_name = prefix + '__' + directory_name
    model_dir = '{}/{}'.format(checkpoint_dir, directory_name)
    return model_dir
