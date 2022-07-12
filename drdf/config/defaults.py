import os.path as osp

from fvcore.common.config import CfgNode

_C = CfgNode()
_C.MATTERPORT_PATH = ""
_C.MATTERPORT_TAR_PATH = ""
_C.DATALOADER = CfgNode()
_C.DATALOADER.LOAD_FROM_TAR = False
_C.DATALOADER.FILTER_DEPTH = True
_C.DATALOADER.SAMPLING = CfgNode()
_C.DATALOADER.SAMPLING.NOISE_VARIANCE = 0.1

## "X" -- xaxis , "Y" -- axis, "C" -- camera direction. All directions are relative to camera.
_C.DATALOADER.SAMPLING.RAY_DIR_LST = ["Y", "C"]
## same length as ray_dir, corresponding to every direction
_C.DATALOADER.SAMPLING.N_RAYS_LST = [20, 20]

_C.DATALOADER.SAMPLING.Z_MAX = 8.0
_C.DATALOADER.SPLIT = "train"
_C.DATALOADER.DATASET_TYPE = "matterport"

_C.DATALOADER.SIGNED_RAY_DIST = True
_C.DATALOADER.UNSIGNED_RAY_DIST = False
_C.DATALOADER.CLAMP_MAX_DIST = 1.0
_C.DATALOADER.IMG_SIZE = 256

_C.DATALOADER.SINGLE_INSTANCE = False
_C.DATALOADER.SINGLE_BATCH = False

_C.MODEL = CfgNode()

_C.MODEL.APPLY_LOG_TRANSFORM = True
_C.MODEL.USE_POINT_FEATURES = True
_C.MODEL.DECODER = "pixNerf"
_C.MODEL.DIR_ENCODING = True
_C.MODEL.RESOLUTION = 128
_C.MODEL.MLP_ACTIVATION = ""

# _C.MODEL.VOXEL.VOXEL_SIZE_Z = 128
# _C.MODEL.VOXEL_HEAD.VOXEL_SIZE_XY = 128

_C.OPTIM = CfgNode()
_C.OPTIM.BETA1 = 0.9
_C.OPTIM.LEARNING_RATE = 0.0003
_C.OPTIM.BETA2 = 0.999
_C.OPTIM.GRAD_CLIPPING = CfgNode()
_C.OPTIM.GRAD_CLIPPING.ENABLED = True
_C.OPTIM.GRAD_CLIPPING.MAX_NORM_VALUE = 1.0

_C.TRAIN = CfgNode()
_C.TRAIN.NUM_EPOCHS = 100
_C.TRAIN.NUM_PRETRAIN_EPOCHS = 0
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.NUM_WORKERS = 8
_C.TRAIN.BN_OFF_EPOCH = 50

_C.TEST = CfgNode()
_C.TEST.NUM_EPOCHS = 200
_C.TEST.BATCH_SIZE = 1
_C.CHECKPOINT_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../../", "cachedir", "checkpoints"
)
_C.LOGGING_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../../", "cachedir", "logs"
)
_C.TENSORBOARD_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../../", "cachedir", "tb_logs"
)
_C.ENABLE_ELASTIC_CHECKPOINTING = True
_C.RESULT_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../../", "cachedir", "results"
)

_C.RENDERER = CfgNode()
_C.RENDERER.RENDER_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)), "../../", "cachedir", "render_dir"
)
_C.RENDERER.RENDER_FUNC = "render_utils.render_mesh_matterport"

_C.NAME = "experiment_name"
_C.ENV_NAME = "main"

_C.LOGGING = CfgNode()
_C.LOGGING.PLOT_SCALARS = True
_C.LOGGING.VISUAL_COUNT = 2
_C.LOGGING.PRINT_FREQ = 10
_C.LOGGING.VALID_EPOCH_FREQ = 1
_C.LOGGING.SAVE_VIS_FREQ = 100
_C.LOGGING.SAVE_CHECKPOINT_FREQ = 1000
_C.LOGGING.SAVE_EPOCH_FREQ = 20
_C.LOGGING.SAVE_VIS = True
_C.LOGGING.WEB_VIS_SERVER = "http://fouheylab.eecs.umich.edu"
_C.LOGGING.WEB_VIS_PORT = 8097


def get_cfg_defaults() -> CfgNode:
    return _C.clone()


curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, "../", "cachedir")


def update_derived_params(cfg):
    # base_data_dir = cfg.BASE_DATA_DIR
    cfg.RESULT_DIR = osp.join(cache_path, "results")
    cfg.CHECKPOINT_DIR = osp.join(cache_path, "checkpoints")
    cfg.LOGGING_DIR = osp.join(cache_path, "logs")
    cfg.RESULTS_DIR = osp.join(cache_path, "results")
    cfg.RENDER_DIR = osp.join(cache_path, "render_dir")

    cfg.TENSORBOARD_DIR = osp.join(cache_path, "tb_logs")
    return cfg
