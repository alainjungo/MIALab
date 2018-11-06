import argparse

import pymia.deeplearning.logging as log

import mialab.configuration.config as cfg
import mialab.data.handler as hdlr
import mialab.data.split as split
import mialab.model.factory as mdl
import mialab.utilities.filesystem as fs
import mialab.utilities.training as train


def main(config_file: str):
    config = cfg.load(config_file, cfg.Configuration)

    # set up directories and logging
    model_dir, result_dir = fs.prepare_directories(config_file, cfg.Configuration,
                                                   lambda: fs.get_directory_name(config))
    config.model_dir = model_dir
    config.result_dir = result_dir
    print(config)

    # load train and valid subjects from split file (also test but it is unused)
    subjects_train, subjects_valid = split.load_split(config.split_file)
    print('Train subjects:', subjects_train)
    print('Valid subjects:', subjects_valid)

    # set up data handling
    data_handler = hdlr.SliceWiseDataHandler(config, subjects_train, subjects_valid, None)

    # extract a sample for model initialization
    data_handler.dataset.set_extractor(data_handler.extractor_train)
    data_handler.dataset.set_transform(data_handler.extraction_transform_train)
    sample = data_handler.dataset[0]

    model = mdl.get_model(config)(sample, config)
    logger = log.TorchLogger(config.model_dir,
                             model.epoch_summaries(), model.batch_summaries(), model.visualization_summaries())

    trainer = train.SegmentationTrainer(data_handler, logger, config, model)
    trainer.train()


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Deep learning for magnetic resonance fingerprinting')

    parser.add_argument(
        '--config_file',
        type=str,
        default='./config/config.json',
        help='Path to the configuration file.'
    )

    args = parser.parse_args()
    main(args.config_file)
