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
STRICT_SHAPES = bool(int(os.environ.get("STRICT_SHAPES", "0")))

H = _env("H", 512, int)
W = _env("W", 512, int)

THRESH = _env("THRESH", 0.0, float)

# Model (keep minimal; pick a preset backbone + embedding width)
BACKBONE = os.environ.get("BACKBONE", "small")   # {tiny, small, base, wide}
EMBED_DIM = _env("EMBED_DIM", 256, int)
SHARD_EVENTS = _env("SHARD_EVENTS", 2048, int)
CHUNK_EVENTS = _env("CHUNK_EVENTS", 256, int)
MAX_BAD_EVENT_LOG = _env("MAX_BAD_EVENT_LOG", 25, int)

UPROOT_DECOMP_WORKERS = _env("UPROOT_DECOMP_WORKERS", 2, int)
FAULTHANDLER_TIMEOUT = _env("FAULTHANDLER_TIMEOUT", 120, int)

# One canonical shards directory (used by both processing + training)
SHARDS_DIR = os.environ.get("SHARDS_DIR", os.environ.get("PROCESS_OUT_DIR", "shards"))
# Backward-compatible alias
PROCESS_OUT_DIR = SHARDS_DIR
SEED = _env("SEED", 123, int)
BATCH_SIZE = _env("BATCH_SIZE", 32, int)
NUM_WORKERS = _env("NUM_WORKERS", 4, int)
MAX_STEPS = _env("MAX_STEPS", 200_000, int)
LR0 = _env("LR0", 0.01, float)
WEIGHT_DECAY = _env("WEIGHT_DECAY", 1e-4, float)
MOMENTUM = _env("MOMENTUM", 0.9, float)
POLY_POWER = _env("POLY_POWER", 0.9, float)
VAL_FRACTION = _env("VAL_FRACTION", 0.1, float)
VAL_EVERY = _env("VAL_EVERY", 2000, int)
VAL_BATCHES = _env("VAL_BATCHES", 50, int)
CHECKPOINT_EVERY = _env("CHECKPOINT_EVERY", 1000, int)
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoint.pt")
