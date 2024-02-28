"""Neural network module."""

from sbijax._src.nn.continuous_normalizing_flow import CCNF, make_ccnf
from sbijax._src.nn.make_flows import (
    make_affine_maf,
    make_surjective_affine_maf,
)
from sbijax._src.nn.make_resnet import make_resnet
from sbijax._src.nn.make_snass_networks import make_snass_net, make_snasss_net
