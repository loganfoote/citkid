import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

################################################################################
############################### test rfmux import ##############################
################################################################################
def test_rfmux_import(monkeypatch):
    """Test that the rfmux module can be imported."""
    try:
        import rfmux 
    except ImportError as e:
        raise ImportError("rfmux module could not be imported. "
                          "Ensure that rfmux dependencies are installed.") from e
    # computer dependent: fails if rfmux dependencies are not working

################################################################################
######################### Fake rfmux module for testing ########################
################################################################################
def _install_fake_rfmux(monkeypatch):
    """Provide a minimal fake rfmux module for import-time decorators."""
    def macro(_cls, register=False):
        def decorator(func):
            return func
        return decorator

    fake_rfmux = SimpleNamespace(
        macro=macro,
        ReadoutModule=type("ReadoutModule", (), {}),
    )
    monkeypatch.setitem(sys.modules, "rfmux", fake_rfmux)

@pytest.fixture()
def instrument_module(monkeypatch):
    _install_fake_rfmux(monkeypatch)

    from citkid.crs import instrument
    return importlib.reload(instrument)

################################################################################
#################################### set_nco ###################################
################################################################################
@pytest.mark.asyncio
async def test_set_nco_uses_module_index(instrument_module):
    device = SimpleNamespace(set_nco_frequency=AsyncMock())
    module = SimpleNamespace(crs=device, module=2)

    nco_freq_dict = {1: 123.0, 2: 456.0, 3: 789.0}
    await instrument_module.set_nco(module, nco_freq_dict)

    device.set_nco_frequency.assert_awaited_once_with(456.0, module=2)


@pytest.mark.asyncio
async def test_set_nco_passes_through_value(instrument_module):
    device = SimpleNamespace(set_nco_frequency=AsyncMock())
    module = SimpleNamespace(crs=device, module=3)

    nco_freq_dict = {3: 1.2345e6}
    await instrument_module.set_nco(module, nco_freq_dict)

    device.set_nco_frequency.assert_awaited_once_with(1.2345e6, module=3)


@pytest.mark.asyncio
async def test_set_nco_missing_module_key_raises(instrument_module):
    device = SimpleNamespace(set_nco_frequency=AsyncMock())
    module = SimpleNamespace(crs=device, module=5)

    nco_freq_dict = {3: 1000}
    with pytest.raises(KeyError):
        await instrument_module.set_nco(module, nco_freq_dict)

    device.set_nco_frequency.assert_not_awaited()