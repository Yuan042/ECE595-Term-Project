from .registry import MODELS

# Supervised Monocular Depth Network
# from .trainers.mono_depth.DORN import DORN
# from .trainers.mono_depth.BTS import BTS
# from .trainers.mono_depth.AdaBins import AdaBins
# from .trainers.mono_depth.Midas import Midas
# from .trainers.mono_depth.NewCRF import NewCRF
# from .trainers.mono_depth.LiteMono import LiteMono

# --- Supervised Stereo Matching Network ---
# Load MonoStereoCRF first so others failing won't block it
try:
    from .trainers.stereo_depth.MonoStereoCRF import MonoStereoCRF
except Exception as e:
    print(f"[models] MonoStereoCRF not loaded: {e}")

try:
    from .trainers.stereo_depth.LiteMono_crf import LiteMono_crf
except Exception as e:
    print(f"[models] MonoStereoCRF not loaded: {e}")
# from .trainers.stereo_depth.MonoStereoCRF import MonoStereoCRF

# The rest are optional; fail-soft
# for mod_name in ["PSMNet", "AANet", "GWCNet", "CFNet", "ACVNet"]:
#     try:
#         module = __import__(
#             f".trainers.stereo_depth.{mod_name}", fromlist=[mod_name])
#         globals()[mod_name] = getattr(module, mod_name)
#     except Exception as e:
#         print(f"[models] {mod_name} not loaded: {e}")
