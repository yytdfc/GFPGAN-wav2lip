# flake8: noqa
import sys
import os.path as osp
from basicsr.train import train_pipeline

import gfpgan.archs
import gfpgan.data
import gfpgan.models


if __name__ == '__main__':
    # print(sys.argv)
    # for i in range(len(sys.argv)):
        # if sys.argv[i].startswith("--local-rank"):
            # sys.argv[i] = "--local_rank" + sys.argv[i][12:]
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
