import pytest

# Skip tests if rfmux is not available
# pytest.importorskip(
#     "rfmux",
#     reason=(
#         "rfmux module could not be imported. "
#         "Ensure that rfmux dependencies are installed."
#     ),
# )

# @pytest.mark.usefixtures("rfmux_imports")
# class TestRfmuxDependent:
#     def test_template(self):
#         raise Exception("This is a test exception")
