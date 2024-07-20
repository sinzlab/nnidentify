from mei.methods import gradient_ascent as ga

from .optimization import MEIWithValidation


def gradient_ascent(*args, **kwargs):
    return ga(mei_class=MEIWithValidation, *args, **kwargs)
