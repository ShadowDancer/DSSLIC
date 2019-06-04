import collections

import yaml


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
