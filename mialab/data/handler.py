import numpy as np
import pymia.data.extraction as pymia_extr
import pymia.data.transformation as pymia_tfm
import pymia.deeplearning.conversion as pymia_cnv
import pymia.deeplearning.data_handler as hdlr

import mialab.configuration.config as cfg


class SliceWiseDataHandler(hdlr.DataHandler):

    def __init__(self, config: cfg.Configuration,
                 subjects_train,
                 subjects_valid,
                 subjects_test,
                 collate_fn=pymia_cnv.TorchCollate(('images', 'labels', 'mask_fg', 'mask_t1h2o'))):
        super().__init__()

        indexing_strategy = pymia_extr.SliceIndexing()

        self.dataset = pymia_extr.ParameterizableDataset(config.database_file,
                                                         indexing_strategy,
                                                         pymia_extr.SubjectExtractor(),  # for the usual select_indices
                                                         None)

        self.no_subjects_train = len(subjects_train)
        self.no_subjects_valid = len(subjects_valid)
        self.no_subjects_test = 0

        # get sampler ids by subjects
        sampler_ids_train = pymia_extr.select_indices(self.dataset, pymia_extr.SubjectSelection(subjects_train))
        sampler_ids_valid = pymia_extr.select_indices(self.dataset, pymia_extr.SubjectSelection(subjects_valid))

        # define extractors
        self.extractor_train = pymia_extr.ComposeExtractor(
            [pymia_extr.DataExtractor(categories=('images', 'labels')),
             pymia_extr.IndexingExtractor(),  # for SubjectAssembler (assembling)
             pymia_extr.ImageShapeExtractor()  # for SubjectAssembler (shape)
             ])

        self.extractor_valid = pymia_extr.ComposeExtractor(
            [pymia_extr.DataExtractor(categories=('images', 'labels')),
             pymia_extr.IndexingExtractor(),  # for SubjectAssembler (assembling)
             pymia_extr.ImageShapeExtractor()  # for SubjectAssembler (shape)
             ])

        self.extractor_test = pymia_extr.ComposeExtractor(
            [pymia_extr.SubjectExtractor(),
             pymia_extr.DataExtractor(categories=('labels', )),
             pymia_extr.ImagePropertiesExtractor(),
             pymia_extr.ImageShapeExtractor()
             ])

        # define transforms for extraction
        self.extraction_transform_train = pymia_tfm.ComposeTransform(
            [pymia_tfm.SizeCorrection((cfg.TENSOR_WIDTH, cfg.TENSOR_HEIGHT)),
             pymia_tfm.Permute((2, 0, 1)),
             pymia_tfm.Squeeze(entries=('labels',), squeeze_axis=0),  # for PyTorch loss functions
             pymia_tfm.LambdaTransform(lambda_fn=lambda np_data : np_data.astype(np.int64), entries=('labels', )),
             # for PyTorch loss functions
             pymia_tfm.ToTorchTensor()])

        self.extraction_transform_valid = pymia_tfm.ComposeTransform(
            [pymia_tfm.SizeCorrection((cfg.TENSOR_WIDTH, cfg.TENSOR_HEIGHT)),
             pymia_tfm.Permute((2, 0, 1)),
             pymia_tfm.Squeeze(entries=('labels',), squeeze_axis=0),  # for PyTorch loss functions
             pymia_tfm.LambdaTransform(lambda_fn=lambda np_data : np_data.astype(np.int64), entries=('labels', )),
             # for PyTorch loss functions
             pymia_tfm.ToTorchTensor()])

        self.extraction_transform_test = None

        # define loaders
        training_sampler = pymia_extr.SubsetRandomSampler(sampler_ids_train)
        self.loader_train = pymia_extr.DataLoader(self.dataset,
                                                  config.batch_size_training,
                                                  sampler=training_sampler,
                                                  collate_fn=collate_fn,
                                                  num_workers=1)

        validation_sampler = pymia_extr.SubsetSequentialSampler(sampler_ids_valid)
        self.loader_valid = pymia_extr.DataLoader(self.dataset,
                                                  config.batch_size_testing,
                                                  sampler=validation_sampler,
                                                  collate_fn=collate_fn,
                                                  num_workers=1)

        self.loader_test = None
