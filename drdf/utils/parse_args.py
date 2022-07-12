import argparse
import pdb
import sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train a Pixel SDF")
    parser.add_argument(
        "--gpu", dest="gpu_id", help="GPU device id to use [0]", default=0, type=int
    )
    parser.add_argument(
        "--cfg", dest="cfg_file", help="optional config file", default=None, type=str
    )
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        help="set config keys",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    results = parser.parse_args()
    return results
