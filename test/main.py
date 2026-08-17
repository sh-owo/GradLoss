import os
import sys

import hydra
from omegaconf import DictConfig

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO, _TEST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train.trainer import run_training


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
