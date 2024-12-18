from yacs.config import CfgNode as CN

_C = CN()

_C.USE_GPU = True
_C.BATCH_SIZE = 32
_C.EPOCHS = 1
_C.EVAL_STEP = 1
_C.SAVE_DIR = 'D:/LVTN'
_C.MODEL_DIR = 'D:/LVTN/models'
_C.NZ = 100
_C.LEARNING_RATE = 0.0003
_C.SEED = 999
_C.NUM_USERS = 55
_C.NUM_EMBEDDING = 128


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
