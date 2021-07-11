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
    def __init__(self, backbone="efficientnet-b0", n_classes=16, MLP_DIM=768, input_shape=[3, 224, 224],
                 HIDDEN_DIM=1280, MAX_POSITION_EMBEDDINGS=288, NUM_ATTENTION_HEADS=10,
                 NUM_HIDDEN_LAYERS=3, ATTENTION_MODE='sliding_chunks', PAD_TOKEN_ID=-1,
                 ATTENTION_WINDOW=[18, 18, 18], INTERMEDIATE_SIZE=3072, ATTENTION_PROBS_DROPOUT_PROB=0.1,
                 HIDDEN_DROPOUT_PROB=0.1, MLP_DROPOUT_RATE=0.4):
        self.input_shape = tuple(input_shape)


if __name__ == '__main__':
    opt = VTNOptions()
    print(opt.backbone)
