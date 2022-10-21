# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from typing import Callable, Dict


CONFIG_REGISTRY = {}

def register_config(config_fn: Callable):
    CONFIG_REGISTRY[config_fn.__name__] = config_fn()
    return config_fn

def get_config(config_str: str) -> Dict:
    """Retrieve composable configs (dictionaries) with period-delimited names.
    Ex: DEFAULT.OPTIONA.OPTIONB will first retrieve DEFAULT then override
    with the specific configs named OPTIONA and OPTIONB.

    Args:
        config_str (str): period delimited, composable config names

    Returns:
        Dict: composed configs.
    """
    config_names = config_str.split(".")
    
    config = {}
    for config_name in config_names:
        config = {**config, **CONFIG_REGISTRY[config_name]}
        
    return config

def merge_args_into_config(args, config):
    """Merge a Namespace args object with the config dictionary. 
    All config attributes that are defined in args will be overridden by the args value.

    Args:
        args (argparse.Namespace): argparse args.
        config (dict): config dict

    Returns:
        dict: config dict
    """
    args_dict = args.__dict__
    for key in config:
        if key in args_dict and args_dict[key] is not None:
            config[key] = args_dict[key]
    return config