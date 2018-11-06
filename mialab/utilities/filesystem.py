import datetime
import os
import shutil

import mialab.configuration.config as cfg


def prepare_epoch_result_directory(result_dir: str, epoch: int) -> str:
    """Creates a result directory named by the epoch on the filesystem.

    Args:
        result_dir (str): The root result directory.
        epoch (int): The epoch number.

    Returns:
        str: The path to the epoch result directory.
    """
    epoch_result_dir = os.path.join(result_dir, 'epoch_{:03d}'.format(epoch))
    os.makedirs(epoch_result_dir, exist_ok=True)
    return epoch_result_dir


def prepare_directories(config_file, config_cls, directory_name_fn) -> (str, str):
    """Prepares the directories for an experiment.

    Args:
        config_file: The config file path.
        config_cls: The config file class.
        directory_name_fn: Lambda to a function that returns a the name for the directories to create, i.e. a str.

    Returns:
        A tuple with the paths to the created model and result directories.
    """
    config = cfg.load(config_file, config_cls)

    # create required directories
    suffix = directory_name_fn()
    model_dir = os.path.join(config.model_dir, suffix)
    result_dir = os.path.join(config.result_dir, suffix)

    # create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # copy config file
    shutil.copyfile(config_file, os.path.join(result_dir, os.path.basename(config_file)))
    shutil.copyfile(config_file, os.path.join(model_dir, os.path.basename(config_file)))

    # copy split file
    if os.path.exists(config.split_file):
        shutil.copyfile(config.split_file, os.path.join(result_dir, os.path.basename(config.split_file)))
        shutil.copyfile(config.split_file, os.path.join(model_dir, os.path.basename(config.split_file)))

    return model_dir, result_dir


def get_directory_name(config: cfg.Configuration, additional: str=None, with_datetime=False) -> str:
    """Gets a directory name to identify the experiment."""

    suffix = config.model
    if config.experiment:
        suffix += '_' + config.experiment

    suffix += '_{}_{}_{}'.format(config.epochs, config.batch_size_training, config.learning_rate)

    if additional:
        suffix += '_' + additional
    if with_datetime:
        suffix += '_' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return suffix
