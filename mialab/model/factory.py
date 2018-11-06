import mialab.configuration.config as cfg
import mialab.model.unet as unet


MODEL_UNKNOWN_ERROR_MESSAGE = 'Unknown model "{}".'


def get_model(config: cfg.Configuration):
    if config.model == unet.MODEL_UNET:
        return unet.UNET
    else:
        raise ValueError(MODEL_UNKNOWN_ERROR_MESSAGE.format(config.model))
