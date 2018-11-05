import json
import os


def split_subjects_portion(subjects: list, proportions: tuple) -> tuple:
    if sum(proportions) != 1.0:
        raise ValueError('portions must sum up to 1')

    nb_total = len(subjects)
    nb_train = int(nb_total * proportions[0])
    nb_valid = int(nb_total * proportions[1])

    train_subjects = subjects[:nb_train]
    valid_subjects = subjects[nb_train:nb_train+nb_valid]

    ret = [train_subjects, valid_subjects]
    return tuple(ret)


def save_split(file_path: str, train_subjects: list, valid_subjects: list):
    if os.path.isfile(file_path):
        os.remove(file_path)

    write_dict = {'train': train_subjects, 'valid': valid_subjects}

    with open(file_path, 'w') as f:
        json.dump(write_dict, f)


def load_split(file_path: str):
    with open(file_path, 'r') as f:
        read_dict = json.load(f)

    train_subjects, valid_subjects = read_dict['train'], read_dict['valid']
    return train_subjects, valid_subjects


