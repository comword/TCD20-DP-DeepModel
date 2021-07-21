from .vtn_effnet import VTN
from .options import VTNOptions


class VTNBuilder:
    def __new__(cls, *args, **kwargs):
        opt = VTNOptions(*args, **kwargs)
        return VTN(opt)

