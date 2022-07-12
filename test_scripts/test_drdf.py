import os.path as osp
import random

import numpy as np
import torch

from drdf.config import defaults
from drdf.test.scene_test import SceneTest
from drdf.utils import parse_args

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    cmd_args = parse_args.parse_args()
    cfg = defaults.get_cfg_defaults()
    if cmd_args.cfg_file is not None:
        cfg.merge_from_file(cmd_args.cfg_file)
    if cmd_args.set_cfgs is not None:
        cfg.merge_from_list(cmd_args.set_cfgs)

    cfg.RESULT_DIR = osp.join(cfg.RESULT_DIR, f"{cfg.NAME}")
    # ray.init(local_mode=False, num_cpus=4)
    tester = SceneTest(cfg)
    tester.initialize()
    tester.test()
