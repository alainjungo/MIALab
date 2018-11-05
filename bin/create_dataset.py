import argparse
import enum
import os
import typing

import SimpleITK as sitk
import numpy as np

import pymia.data as pymia_data
import pymia.data.conversion as conv
import pymia.data.creation as pymia_crt
import pymia.data.loading as pymia_load
import pymia.data.transformation as pymia_tfm
import pymia.data.creation.fileloader as file_load


class FileTypes(enum.Enum):
    T1 = 1  # The T1-weighted image
    T2 = 2  # The T2-weighted image
    GT = 3  # The ground truth image


class LoadData(file_load.Load):

    def __call__(self, file_name: str, id_: str, category: str, subject_id: str) -> \
            typing.Tuple[np.ndarray, typing.Union[conv.ImageProperties, None]]:
        if category == 'images':
            img = sitk.ReadImage(file_name, sitk.sitkFloat32)
        else:
            img = sitk.ReadImage(file_name, sitk.sitkUInt8)

        return sitk.GetArrayFromImage(img), conv.ImageProperties(img)


class Subject(pymia_data.SubjectFile):

    def __init__(self, subject: str, files: dict):
        super().__init__(subject,
                         images={FileTypes.T1.name: files[FileTypes.T1], FileTypes.T2.name: files[FileTypes.T2]},
                         labels={FileTypes.GT.name: files[FileTypes.GT]})
        self.subject_path = files.get(subject, '')


class DataSetFilePathGenerator(pymia_load.FilePathGenerator):
    """Represents a brain image file path generator.

    The generator is used to convert a human readable image identifier to an image file path,
    which allows to load the image.
    """

    def __init__(self):
        """Initializes a new instance of the DataSetFilePathGenerator class."""
        pass

    @staticmethod
    def get_full_file_path(id_: str, root_dir: str, file_key, file_extension: str) -> str:
        """Gets the full file path for an image.
        Args:
            id_ (str): The image identification.
            root_dir (str): The image' root directory.
            file_key (object): A human readable identifier used to identify the image.
            file_extension (str): The image' file extension.
        Returns:
            str: The images' full file path.
        """
        add_file_extension = True

        if file_key == FileTypes.T1:
            file_name = 'T1mni_biasfieldcorr_noskull'
        elif file_key == FileTypes.T2:
            file_name = 'T2mni_biasfieldcorr_noskull'
        elif file_key == FileTypes.GT:
            file_name = 'labels_mniatlas'
        else:
            raise ValueError('Unknown key')

        file_name = file_name + file_extension if add_file_extension else file_name
        return os.path.join(root_dir, file_name)


class DirectoryFilter(pymia_load.DirectoryFilter):
    """Represents a data directory filter."""

    def __init__(self):
        """Initializes a new instance of the DataDirectoryFilter class."""
        pass

    @staticmethod
    def filter_directories(dirs: typing.List[str]) -> typing.List[str]:
        """Filters a list of directories.
        Args:
            dirs (List[str]): A list of directories.
        Returns:
            List[str]: The filtered list of directories.
        """

        # currently, we do not filter the directories. but you could filter the directory list like this:
        # return [dir for dir in dirs if not dir.lower().__contains__('atlas')]
        return sorted(dirs)


def main(hdf_file: str, data_dir: str):
    keys = [FileTypes.T1, FileTypes.T2, FileTypes.GT]
    crawler = pymia_load.FileSystemDataCrawler(data_dir,
                                               keys,
                                               DataSetFilePathGenerator(),
                                               DirectoryFilter(),
                                               '.nii.gz')

    subjects = [Subject(id_, file_dict) for id_, file_dict in crawler.data.items()]

    if os.path.exists(hdf_file):
        os.remove(hdf_file)

    with pymia_crt.get_writer(hdf_file) as writer:
        callbacks = pymia_crt.get_default_callbacks(writer)

        # normalize the images and unsqueeze the labels and mask.
        # Unsqueeze is needed due to the convention to have the number of channels as last dimension.
        # I.e., here we have the shape 10 x 256 x 256 before the unsqueeze operation and after 10 x 256 x 256 x 1
        transform = pymia_tfm.ComposeTransform([pymia_tfm.IntensityNormalization(loop_axis=3, entries=('images',)),
                                                pymia_tfm.UnSqueeze(entries=('labels',))
                                                ])

        traverser = pymia_crt.SubjectFileTraverser()
        traverser.traverse(subjects, callback=callbacks, load=LoadData(), transform=transform)


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Dataset creation')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../data/train_dataset.h5',
        # default='../data/test_dataset.h5',
        help='Path to the dataset file.'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data/train',
        # default='../data/test',
        help='Path to the data directory.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.data_dir)
