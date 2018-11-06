import pymia.config.configuration as cfg
import pymia.deeplearning.config as dlcfg


# Some configuration variables, which most likely do not change.
# Therefore, they are not added to the Configuration class
TENSOR_WIDTH = 200  # network input width
TENSOR_HEIGHT = 160  # network input height
NO_CLASSES = 6  # number of classes or network input channels


class Configuration(dlcfg.DeepLearningConfiguration):
    """Represents a configuration."""

    VERSION = 1
    TYPE = ''

    @classmethod
    def version(cls) -> int:
        return cls.VERSION

    @classmethod
    def type(cls) -> str:
        return cls.TYPE

    def __init__(self):
        """Initializes a new instance of the Configuration class.

        Args:
            config (dict): A dictionary representing the configuration.
        """
        super().__init__()
        self.split_file = ''

        self.model = ''  # string identifying the model
        self.experiment = ''  # string to describe experiment

        # training configuration
        self.learning_rate = 0.01  # the learning rate
        self.dropout_p = 0.2

        # network configuration
        self.n_pooling = 3
        self.n_channels = 32


def load(path: str, config_cls):
    """Loads a configuration file.

    Args:
        path (str): The path to the configuration file.
        config_cls (class): The configuration class (not an instance).

    Returns:
        (config_cls): The configuration.
    """

    return cfg.load(path, config_cls)
