from .version import __version__, __versiondate__, __license__
# Import the actual model

from .pinn.base import PINN
from .pinn.wave import PINN_WAVE
from .pinn.separated import PINN_SEP
from sibpinn.config_gpu import config_gpu
from sibpinn.utils import from_file