"""
general configs for source2source
"""

# In case we want to have separate config files for Dev/Staging/Live environments in the future
ENV = "dev"

# Path to the `.py` config files for the project
PATH = "configs/"

# Some global variables to hold metadata
CONVERSION_OPTIONS = "not_s2s"
MAX_TRACED_PROGRAM_COUNT = 10
BASE_OUTPUT_DIR = "ivy_transpiled_outputs/"

# Frameworks whose code S2S supports currently
SUPPORTED_S2S_SOURCES = ["torch", "ivy"]
SUPPORTED_S2S_TARGETS = ["tensorflow", "jax", "numpy", "ivy"]
