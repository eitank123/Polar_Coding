import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils import available_cpus  # noqa: E402

torch.set_num_threads(available_cpus())


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end / statistical tests (deselect with -m 'not slow')")


@pytest.fixture(autouse=True)
def _seed():
    from sionna.phy import config as sionna_config
    sionna_config.seed = 1234


@pytest.fixture(scope="session")
def link128():
    from polar import PolarLink
    return PolarLink(512, 128, 8, "CRC6")
