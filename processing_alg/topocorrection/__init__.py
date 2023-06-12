#!/usr/bin/env python
""" Terraform QGIS plugin.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = 'Tigran Manasyan'
__copyright__ = '(C) 2023 by Tigran Manasyan'
__license__ = "GPLv3"

from .CTopoCorrectionAlgorithm import CTopoCorrectionAlgorithm
from .CosineCTopoCorrectionAlgorithm import CosineCTopoCorrectionAlgorithm
from .CosineTTopoCorrectionAlgorithm import CosineTTopoCorrectionAlgorithm
from .MinnaertScsTopoCorrectionAlgorithm import MinnaertScsTopoCorrectionAlgorithm
from .MinnaertTopoCorrectionAlgorithm import MinnaertTopoCorrectionAlgorithm
from .PbcTopoCorrectionAlgorithm import PbcTopoCorrectionAlgorithm
from .PbmTopoCorrectionAlgorithm import PbmTopoCorrectionAlgorithm
from .ScsCTopoCorrectionAlgorithm import ScsCTopoCorrectionAlgorithm
from .ScsTopoCorrectionAlgorithm import ScsTopoCorrectionAlgorithm
from .TeilletRegressionTopoCorrectionAlgorithm import TeilletRegressionTopoCorrectionAlgorithm
from .VecaTopoCorrectionAlgorithm import VecaTopoCorrectionAlgorithm

DEFAULT_CORRECTIONS = [
    CosineTTopoCorrectionAlgorithm,
    CosineCTopoCorrectionAlgorithm,
    CTopoCorrectionAlgorithm,
    ScsTopoCorrectionAlgorithm,
    ScsCTopoCorrectionAlgorithm,
    MinnaertTopoCorrectionAlgorithm,
    MinnaertScsTopoCorrectionAlgorithm,
    PbmTopoCorrectionAlgorithm,
    VecaTopoCorrectionAlgorithm,
    TeilletRegressionTopoCorrectionAlgorithm,
    PbcTopoCorrectionAlgorithm
]
