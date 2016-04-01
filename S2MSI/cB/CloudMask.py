from .cB import read_classical_Bayesian_persistence_file
from .cB import get_clf_functions
from .cB import ToClassifierDef
from .cB import S2cB
from .cB import ClassicalBayesian

from pkg_resources import resource_filename, Requirement, DistributionNotFound
from os.path import dirname, join, isfile
import logging


class CloudMask(S2cB):
    def __init__(self,persistence_file=None,processing_tiles=10,logger=None):
        """ Get Cloud Detection based on classical Bayesian approach

        :param persistence_file: if None, use internal file, else give file name to persistence file
        :param processing_tiles: in order so save memory, the processing can be done in tiles
        :param logger: None or logger instance
        :return: CloudMask instance
        """

        logger = logger or logging.getLogger(__name__)

        if persistence_file is None:
            persistence_file = "data/cld_mask_20160321_s2.h5"
            try:
                fn = resource_filename(Requirement.parse("S2MSI"), persistence_file)
            except DistributionNotFound:
                fn = join(dirname(__file__), persistence_file)
                if isfile(fn) is False:
                    raise FileNotFoundError(persistence_file)
            else:
                if isfile(fn) is False:
                    fn = join(dirname(__file__), persistence_file)
                    if isfile(fn) is False:
                        raise FileNotFoundError(persistence_file)
            self.persistence_file = fn
        else:
            self.persistence_file = persistence_file

        data = read_classical_Bayesian_persistence_file(filename=self.persistence_file)
        cb_clf = ClassicalBayesian(mk_clf=ToClassifierDef(clf_functions=get_clf_functions(),**data["kwargs_mk_clf"]),
                                   **data["kwargs_cB"])

        super().__init__(cb_clf=cb_clf,
                         mask_legend=data["mask_legend"],
                         clf_to_col=data["clf_to_col"],
                         processing_tiles=processing_tiles,
                         logger=logger)
