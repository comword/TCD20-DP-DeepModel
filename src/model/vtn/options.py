from functools import wraps
import inspect


def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    argspec = inspect.getfullargspec(func)
    names = argspec.args
    defaults = argspec.defaults

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


class VTNOptions:
    @initializer
    def __init__(self, backbone="EfficientNetB0", n_classes=16, MLP_DIM=512, input_shape=[224, 224, 3],
                 HIDDEN_DIM=512, MAX_POSITION_EMBEDDINGS=128, NUM_ATTENTION_HEADS=4, NUM_HIDDEN_LAYERS=2, 
                 PAD_TOKEN_ID=-1, INTERMEDIATE_SIZE=1024, ATTENTION_PROBS_DROPOUT_PROB=0.1,
                 HIDDEN_DROPOUT_PROB=0.1, MLP_DROPOUT_RATE=0.4):
        self.input_shape = tuple(input_shape)


if __name__ == '__main__':
    opt = VTNOptions()
    print(opt.backbone)
