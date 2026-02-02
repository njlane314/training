import os


def _env(name, default, cast):
    """
    @brief Read and cast an environment variable with a default.
    """
    v = os.environ.get(name)
    return default if v is None else cast(v)


ROOT_FILE = os.environ.get("ROOT_FILE", "/gluster/data/dune/niclane/events.root")
TREE = os.environ.get("TREE", "events")
BR_U = os.environ.get("BR_U", "detector_image_u")
BR_V = os.environ.get("BR_V", "detector_image_v")
BR_W = os.environ.get("BR_W", "detector_image_w")
BR_Y = os.environ.get("BR_Y", "is_signal")
BR_WGT = os.environ.get("BR_WGT", "w_nominal")

SHARDS_DIR = os.environ.get("SHARDS_DIR", "/gluster/data/dune/niclane/sparse_shards")
SHARDS_OUT = os.environ.get("SHARDS_OUT", SHARDS_DIR)
PROCESS_OUT_DIR = os.environ.get("PROCESS_OUT_DIR", "shards")

H = _env("H", 512, int)
W = _env("W", 512, int)

THRESH = _env("THRESH", 0.0, float)
ADC_SIGNLOG = _env("ADC_SIGNLOG", 0, int) != 0

SHARD_EVENTS = _env("SHARD_EVENTS", 2048, int)
CHUNK_EVENTS = _env("CHUNK_EVENTS", 64, int)

BATCH = _env("BATCH", 32, int)
STEPS_PER_EPOCH = _env("STEPS_PER_EPOCH", 1000, int)
EPOCHS = _env("EPOCHS", 20, int)
LR = _env("LR", 3e-3, float)
WEIGHT_DECAY = _env("WEIGHT_DECAY", 1e-4, float)
NUM_WORKERS = _env("NUM_WORKERS", 8, int)
SEED = _env("SEED", 12345, int)
EMA = _env("EMA", 0.98, float)
GRAD_CLIP = _env("GRAD_CLIP", 1.0, float)
GRAD_ACCUM_STEPS = _env("GRAD_ACCUM_STEPS", 1, int)
WARMUP_RATIO = _env("WARMUP_RATIO", 0.02, float)
MIN_LR_RATIO = _env("MIN_LR_RATIO", 0.05, float)
SCHED = _env("SCHED", 0, int) != 0
AMP = _env("AMP", 0, int) != 0
VAL_PROBE_EVERY = _env("VAL_PROBE_EVERY", 1, int)
VAL_FRAC = _env("VAL_FRAC", 0.1, float)
OUT = os.environ.get("OUT", "checkpoint.pt")
LOG_OUT = os.environ.get("LOG_OUT", "training_metrics.log")

BASE_FILTERS = _env("BASE_FILTERS", 32, int)
NUM_STRIDES = _env("NUM_STRIDES", 3, int)
DROPOUT = _env("DROPOUT", 0.2, float)

LLR_SHARDS_DIR = os.environ.get("LLR_SHARDS_DIR", "shards")
LLR_SEED = _env("LLR_SEED", 123, int)
LLR_BATCH_SIZE = _env("LLR_BATCH_SIZE", 16, int)
LLR_NUM_WORKERS = _env("LLR_NUM_WORKERS", 4, int)
LLR_MAX_STEPS = _env("LLR_MAX_STEPS", 200_000, int)
LLR_LR0 = _env("LLR_LR0", 1e-1, float)
LLR_WEIGHT_DECAY = _env("LLR_WEIGHT_DECAY", 1e-4, float)
LLR_MOMENTUM = _env("LLR_MOMENTUM", 0.9, float)
LLR_POLY_POWER = _env("LLR_POLY_POWER", 0.9, float)
LLR_VAL_FRACTION = _env("LLR_VAL_FRACTION", 0.1, float)
LLR_VAL_EVERY = _env("LLR_VAL_EVERY", 2000, int)
LLR_VAL_BATCHES = _env("LLR_VAL_BATCHES", 50, int)
