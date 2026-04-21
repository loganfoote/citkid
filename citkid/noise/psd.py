import warnings

from ..signal.psd import bin_psd, filter_pt, get_csd, get_psd

warnings.warn(
    "citkid.noise.psd is deprecated. "
    "Please use citkid.signal.psd instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["get_psd", "get_csd", "bin_psd", "filter_pt"]
