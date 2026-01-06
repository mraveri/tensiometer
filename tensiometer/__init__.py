"""Package metadata and convenience constants."""

import warnings

__author__ = 'Marco Raveri'
__version__ = "1.1.0"
__url__ = "https://tensiometer.readthedocs.io"

warnings.filterwarnings(
    "ignore",
    message="In the future `np.object` will be defined as the corresponding NumPy scalar.",
    category=FutureWarning,
)
