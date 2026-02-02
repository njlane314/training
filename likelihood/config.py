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

H = _env("H", 512, int)
W = _env("W", 512, int)

THRESH = _env("THRESH", 0.0, float)
SHARD_EVENTS = _env("SHARD_EVENTS", 2048, int)
CHUNK_EVENTS = _env("CHUNK_EVENTS", 64, int)

# One canonical shards directory (used by both processing + training)
SHARDS_DIR = os.environ.get("SHARDS_DIR", os.environ.get("PROCESS_OUT_DIR", "shards"))
# Backward-compatible alias
PROCESS_OUT_DIR = SHARDS_DIR
SEED = _env("SEED", 123, int)
BATCH_SIZE = _env("BATCH_SIZE", 16, int)
NUM_WORKERS = _env("NUM_WORKERS", 4, int)
MAX_STEPS = _env("MAX_STEPS", 200_000, int)
LR0 = _env("LR0", 1e-1, float)
WEIGHT_DECAY = _env("WEIGHT_DECAY", 1e-4, float)
MOMENTUM = _env("MOMENTUM", 0.9, float)
POLY_POWER = _env("POLY_POWER", 0.9, float)
VAL_FRACTION = _env("VAL_FRACTION", 0.1, float)
VAL_EVERY = _env("VAL_EVERY", 2000, int)
VAL_BATCHES = _env("VAL_BATCHES", 50, int)
