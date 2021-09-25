# flake8: noqa

import pkg_resources


__version__ = pkg_resources.get_distribution("srhandnet").version


import srhandnet.data

from srhandnet.peak import peak_local_max
from srhandnet.model import SRHandNet
