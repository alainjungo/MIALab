import argparse
import os

import mialab.data.split as split


def main():

    training_data_dir = '../data/train'
    out_split_path = '../data/split.json'

    subject_names = os.listdir(training_data_dir)

    train_names, validation_names = split.split_subjects_portion(subject_names, (0.8, 0.2))
    split.save_split(out_split_path, train_names, validation_names)


if __name__ == '__main__':
    main()
