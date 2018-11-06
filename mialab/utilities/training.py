import os

import numpy as np
import pymia.data.assembler as asmbl
import pymia.data.conversion as conv
import pymia.data.transformation as tfm
import pymia.deeplearning.training as train
import pymia.deeplearning.logging as log
import pymia.deeplearning.model as mdl
import SimpleITK as sitk

import mialab.configuration.config as cfg
import mialab.data.handler as hdlr
import mialab.utilities.evaluation as eval
import mialab.utilities.filesystem as fs


def size_correction(params: dict):
    data = params['__prediction']
    idx = params['batch_idx']
    batch = params['batch']

    data = np.transpose(data, (1, 2, 0))  # transpose back from PyTorch convention
    data = np.argmax(data, -1)  # convert to class labels

    # correct size
    correct_shape = batch['shape'][idx][1:]
    transform = tfm.SizeCorrection(correct_shape, 0, entries=('data',))
    data = transform({'data': data})['data']

    data = np.expand_dims(data, -1)  # for dataset convention
    return data, batch

def init_shape(shape, id_):
    shape = list(shape)
    shape[-1] = 1  # do to PyTorchs channel convention
    return np.zeros(shape, np.float32)

def init_subject_assembler():
    return asmbl.SubjectAssembler(zero_fn=init_shape, on_sample_fn=size_correction)


def validate_on_subject_training(self: train.Trainer, subject_assembler: asmbl.SubjectAssembler,
                                 config: cfg.Configuration):
    # prepare filesystem and evaluator
    if self.current_epoch % self.save_validation_nth_epoch == 0:
        epoch_result_dir = fs.prepare_epoch_result_directory(config.result_dir, self.current_epoch)
        epoch_csv_file = os.path.join(
            epoch_result_dir,
            '{}_{}_train.csv'.format(os.path.basename(config.result_dir), self.current_epoch))
        evaluator = eval.init_evaluator(epoch_csv_file)

    else:
        evaluator = eval.init_evaluator(None)

    # loop over all subjects
    print('Epoch {:d}, {} s:'.format(self.current_epoch, self.epoch_duration))
    for subject_idx in list(subject_assembler.predictions.keys()):
        subject_data = self.data_handler.dataset.direct_extract(self.data_handler.extractor_test, subject_idx)
        subject_name = subject_data['subject']
        labels = subject_data['labels']
        labels = np.squeeze(labels, -1)

        prediction = subject_assembler.get_assembled_subject(subject_idx)
        prediction = np.squeeze(prediction, -1)

        evaluator.evaluate(prediction, labels, subject_name)


def validate_on_subject(self: train.Trainer, subject_assembler: asmbl.SubjectAssembler,
                        config: cfg.Configuration) -> float:

    # prepare filesystem and evaluator
    if self.current_epoch % self.save_validation_nth_epoch == 0:
        epoch_result_dir = fs.prepare_epoch_result_directory(config.result_dir, self.current_epoch)
        epoch_csv_file = os.path.join(
            epoch_result_dir,
            '{}_{}.csv'.format(os.path.basename(config.result_dir), self.current_epoch))
        evaluator = eval.init_evaluator(epoch_csv_file)

    else:
        epoch_result_dir = None
        evaluator = eval.init_evaluator(None)

    # loop over all subjects
    print('Epoch {:d}, {} s:'.format(self.current_epoch, self.epoch_duration))
    for subject_idx in list(subject_assembler.predictions.keys()):
        subject_data = self.data_handler.dataset.direct_extract(self.data_handler.extractor_test, subject_idx)
        subject_name = subject_data['subject']
        labels = subject_data['labels']
        labels = np.squeeze(labels, -1)

        prediction = subject_assembler.get_assembled_subject(subject_idx)
        prediction = np.squeeze(prediction, -1)

        evaluator.evaluate(prediction, labels, subject_name)

        # Save predictions as SimpleITK images and save other images
        if self.current_epoch % self.save_validation_nth_epoch == 0:
            subject_results = os.path.join(epoch_result_dir, subject_name)
            os.makedirs(subject_results, exist_ok=True)

            # save predicted maps
            prediction_image = conv.NumpySimpleITKImageBridge.convert(prediction, subject_data['properties'])
            sitk.WriteImage(prediction_image, os.path.join(subject_results, '{}_PREDICTION.mha'.format(subject_name)),
                            True)

    # aggregate results over all subjects
    result_dict = eval.aggregate_results(evaluator)
    score = []

    # log to TensorBoard
    for metric, results_by_label in result_dict.items():
        for label, result in results_by_label.items():
            self.logger.log_scalar('{}/{}-MEAN'.format(label, metric), result[0], self.current_epoch)
            self.logger.log_scalar('{}/{}-STD'.format(label, metric), result[1], self.current_epoch)
        if metric == 'DICE':
            score.append(result[0])  # aggregate mean Dice coefficient for best model

    return float(np.mean(score))


class SegmentationTrainer(train.TorchTrainer):

    def __init__(self, data_handler: hdlr.SliceWiseDataHandler, logger: log.Logger, config: cfg.Configuration,
                 model: mdl.TorchModel):
        super().__init__(data_handler, logger, config, model)
        self.config = config

    def init_subject_assembler(self) -> asmbl.Assembler:
        return init_subject_assembler()

    def validate_on_subject(self, subject_assembler: asmbl.SubjectAssembler, is_training: bool) -> float:
        if is_training:
            validate_on_subject_training(self, subject_assembler, self.config)
            return -1
        else:
            return validate_on_subject(self, subject_assembler, self.config)
