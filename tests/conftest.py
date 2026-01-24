import os

# Ensure rfmux uses no periscope during test collection/imports.
os.environ["CRS_EMBEDDED"] = "1"
