import pymia.deeplearning.model as mdl
import torch.optim as optim
import torch.nn as nn

import mialab.configuration.config as cfg


class TorchMRFModel(mdl.TorchModel):

    def inference(self, x) -> object:
        return self.network(x)

    def loss_function(self, prediction, label=None, **kwargs):
        loss_val = self.loss(prediction, label)
        return loss_val

    def optimize(self, **kwargs):
        self.optimizer.step()

    def __init__(self, sample: dict, config: cfg.Configuration, network):
        super().__init__(config.model_dir, 3)

        self.learning_rate = config.learning_rate
        self.dropout_p = config.dropout_p

        self.network = network(2, cfg.NO_CLASSES, n_channels=config.n_channels, n_pooling=config.n_pooling)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

    def epoch_summaries(self) -> list:
        return []

    def batch_summaries(self):
        return []

    def visualization_summaries(self):
        return []
