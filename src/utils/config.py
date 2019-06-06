import collections
import yaml
import shutil
from os import path
from glob import glob
import re
import tensorflow as tf


def load_config(yml_path, transforms=None):
    """
    :param yml_path: path to yaml file to load
    :param transforms: function that gets yml parsed to dict and may modify it in place (to add defaults etc.)
    :return: yml parsed to class
    """
    with open(yml_path) as f:
        dic = yaml.safe_load(f)

    if transforms is not None:
        transforms(dic)

    return _dict_to_class(dic)


def _dict_to_class(dic):
    if isinstance(dic, collections.Mapping):
        for key, value in dic.items():
            dic[key] = _dict_to_class(value)
        return _namedtuple_from_mapping(dic)
    return dic


def _namedtuple_from_mapping(dic, name="Config"):
    return collections.namedtuple(name, dic.keys())(**dic)


class CopyConfigHook(tf.train.SessionRunHook):
    def __init__(self, config_file, model_dir):
        self.config_copied = False
        self.config_file = config_file
        self.model_dir = model_dir

    def begin(self):

        if not self.config_copied:
            copy_config_to_model_dir(self.config_file, self.model_dir)


def copy_config_to_model_dir(config_path, model_directory):
    shutil.copy(config_path, path.join(model_directory, 'config.yml'))
    existing_cfgs = glob(path.join(model_directory, 'config_*.yml'))


    if len(existing_cfgs) == 0:
        new_cfg_num = 1
    else:

        def extract_num(string):
            res = [int(x) for x in re.findall("config_([0-9]+).yml", string)]
            if len(res) > 0:
                return res
            else:
                return 0
        new_cfg_num = max(extract_num(x) for x in existing_cfgs) or 0 + 1
    shutil.copy(config_path, path.join(model_directory, 'config_' + str(new_cfg_num) + '.yml'))
