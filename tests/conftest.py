import pytest

from liverguard.data.config import DocumentedReference, load_config


@pytest.fixture
def make_cfg():
    """Default config pointed at a test tree, with published-count comparison disabled."""
    def _make(root, **kw):
        cfg = load_config(data_root=root, **kw)
        cfg.documented_reference = DocumentedReference()
        return cfg
    return _make
