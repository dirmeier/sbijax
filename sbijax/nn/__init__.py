"""Functionality to construct neural networks."""

from sbijax._src.nn.make_consistency_model import make_cm
from sbijax._src.nn.make_continuous_flow import make_cnf
from sbijax._src.nn.make_flow import make_maf, make_spf
from sbijax._src.nn.make_mdn import make_mdn
from sbijax._src.nn.make_mlp import make_mlp
from sbijax._src.nn.make_nass_network import make_nass_net, make_nasss_net
from sbijax._src.nn.make_resnet import make_resnet

__all__ = [
    "make_mdn",
    "make_maf",
    "make_spf",
    #
    "make_cnf",
    #
    "make_mlp",
    "make_resnet",
    #
    "make_cm",
    #
    "make_nass_net",
    "make_nasss_net",
]
