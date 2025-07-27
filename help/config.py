# Cell 1 — Imports & Config
import re, math, torch, tqdm, pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download, list_repo_files

from transformer_lens import HookedTransformer, HookedTransformerConfig
from devinterp.optim import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import default_nbeta

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ID    = "timaeus/tetrahedron-3m-og"
CKPT_DIR   = "checkpoints"
KEEP_EVERY = 500
RANGE_START= 500
RANGE_END  = 2500
PERT_FRAC  = 1e-2
