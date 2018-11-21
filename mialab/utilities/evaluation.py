import pymia.evaluation.evaluator as eval
import pymia.evaluation.metric as metric

import numpy as np


def init_evaluator(csv_file: str=None):
    evaluator = eval.Evaluator(eval.ConsoleEvaluatorWriter(5))
    if csv_file is not None:
        evaluator.add_writer(eval.CSVEvaluatorWriter(csv_file))
    evaluator.add_writer(EvaluatorAggregator())
    evaluator.metrics = [metric.DiceCoefficient()]
    evaluator.add_label(1, "WhiteMatter")
    evaluator.add_label(2, "GreyMatter")
    evaluator.add_label(3, "Hippocampus")
    evaluator.add_label(4, "Amygdala")
    evaluator.add_label(5, "Thalamus")
    return evaluator


class EvaluatorAggregator(eval.IEvaluatorWriter):

    def __init__(self):
        self.metrics = {}
        self.results = {}

    def clear(self):
        self.results = {}
        for metric in self.metrics.keys():
            self.results[metric] = {}

    def write(self, data: list):
        """Aggregates the evaluation results.

        Args:
            data (list of list): The evaluation data,
                e.g. [['PATIENT1', 'BACKGROUND', 0.90], ['PATIENT1', 'TUMOR', '0.62']]
        """
        for metric, metric_idx in self.metrics.items():
            for data_item in data:
                if not data_item[1] in self.results[metric]:
                    self.results[metric][data_item[1]] = []
                self.results[metric][data_item[1]].append(data_item[metric_idx])

    def write_header(self, header: list):
        self.metrics = {}
        for metric_idx, metric in enumerate(header[2:]):
            self.metrics[metric] = metric_idx + 2
        self.clear()  # init results dict


class AggregatedResult:

    def __init__(self, label: str, metric: str, mean: float, std: float):
        self.label = label
        self.metric = metric
        self.mean = mean
        self.std = std


def aggregate_results(evaluator: eval.Evaluator) -> typing.List[AggregatedResult]:
    for writer in evaluator.writers:
        if isinstance(writer, EvaluatorAggregator):
            results = []
            for metric in writer.metrics.keys():
                for label, values in writer.results[metric].items():
                    results.append(AggregatedResult(label, metric, float(np.mean(values)), float(np.std(values))))
            writer.clear()
            return results
