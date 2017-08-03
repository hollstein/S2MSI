from six import PY2
__author__ = "Andre Hollstein"

if PY2 is False:  # for python >2
    from .S2Image import S2Image
    from .S2Mask import S2Mask
    from .GranuleInfo import GranuleInfo
    from .GranuleDEM import GranuleDEM
    from .Tools import Tools
    from .cB import CloudMask

    __all__ = ["GranuleInfo",
               "S2Image",
               "S2Mask",
               "GranuleDEM",
               "Tools",
               "GrassTools",
               "CloudMask",
               ]
else:  # for python 2
    from .GrassTools2 import GrassTools

    __all__ = ["GrassTools"]