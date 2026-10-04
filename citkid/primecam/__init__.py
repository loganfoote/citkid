# Legacy code for the Prime-Cam RFSoC readout, kept for existing users. This
# package is meant to be self-contained: the rest of citkid must not import
# from it, and it should not depend on other legacy citkid modules.
#
# instrument.py: stable, tested with mock and (future) hardware tests.
# gain.py, fitter.py, and the fit-row functions in data_io.py: tested in
#   tests/primecam/test_gain_fitter.py.
#
# The following modules intentionally have no tests:
#   procedures.py, update_ares.py, update_fres.py, analysis.py, plot.py,
#   the import functions in data_io.py, and noise/
