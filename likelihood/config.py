import os


def _env(name, default, cast):
    """
    @brief Read and cast an environment variable with a default.
    """
    v = os.environ.get(name)
    return default if v is None else cast(v)


ROOT_FILE = os.environ.get("ROOT_FILE", "/gluster/data/dune/niclane/events.root")
TREE = os.environ.get("TREE", "events")

SHARDS_DIR = os.environ.get("SHARDS_DIR", "/gluster/data/dune/niclane/sparse_shards")
SHARDS_OUT = os.environ.get("SHARDS_OUT", SHARDS_DIR)

H = _env("H", 512, int)
W = _env("W", 512, int)

THRESH = _env("THRESH", 0.0, float)
ADC_SIGNLOG = _env("ADC_SIGNLOG", 0, int) != 0

SHARD_EVENTS = _env("SHARD_EVENTS", 2048, int)
CHUNK_EVENTS = _env("CHUNK_EVENTS", 64, int)

BATCH = _env("BATCH", 32, int)
EPOCHS = _env("EPOCHS", 20, int)
LR = _env("LR", 3e-4, float)
WEIGHT_DECAY = _env("WEIGHT_DECAY", 1e-4, float)
NUM_WORKERS = _env("NUM_WORKERS", 8, int)
SEED = _env("SEED", 12345, int)
EMA = _env("EMA", 0.98, float)
GRAD_CLIP = _env("GRAD_CLIP", 1.0, float)
VAL_FRAC = _env("VAL_FRAC", 0.1, float)
OUT = os.environ.get("OUT", "checkpoint.pt")

BASE_FILTERS = _env("BASE_FILTERS", 32, int)
NUM_STRIDES = _env("NUM_STRIDES", 4, int)
DROPOUT = _env("DROPOUT", 0.2, float)
