from .vtn_effnet import VTN
from .vtn_vit import VTNVIT
from .options import VTNOptions


class VTNBuilder:
    def __new__(cls, *args, **kwargs):
        opt = VTNOptions(*args, **kwargs)
        return VTN(opt)

class VTNVITBuilder:
    def __new__(cls, *args, **kwargs):
        opt = VTNOptions(*args, **kwargs)
        return VTNVIT(opt)
