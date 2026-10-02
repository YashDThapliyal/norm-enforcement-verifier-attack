"""Experiment constants. Values follow the paper (arXiv 2607.09766, App. A.5 / §6) unless noted."""

# Fishery (paper App. A.5)
CARRYING_CAPACITY = 3000
INITIAL_POPULATION = 3000
REGEN_RATE = 0.3
COLLAPSE_THRESHOLD = 100
MAX_HARVEST = 100
WARN_CAUTION_RATIO = 0.4
WARN_COLLAPSE_RATIO = 0.2

# Simulation (paper uses T=15, W=5)
NUM_AGENTS = 8
NUM_ROUNDS = 15  # round 1 is communication-only, rounds 2..15 harvest
HISTORY_WINDOW = 5
VERIFIER_WINDOW = 3
MAX_MESSAGE_WORDS = 200

# Reputation mechanisms (paper §6.2)
REP_ALPHA = 2.0
REP_BETA = 1.0
ESC_K = 3.0

# Models
AGENT_MODEL_LOCAL = "qwen3:8b"
VERIFIER_MODEL = "qwen3:0.6b"
VERIFIER_SIZES = ("qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "qwen3:8b")
JUDGE_MODEL = "claude-haiku-4-5"
VALIDATION_MODEL = "claude-sonnet-5-5"
AGENT_MODEL_API = "claude-haiku-4-5"

AGENT_TEMPERATURE = 0.7
AGENT_MAX_TOKENS = 600
VERIFIER_MAX_TOKENS = 200

# API budget. Prices in USD per million tokens (claude-api reference, cached 2026-09-25).
API_COST_CAP_USD = 9.00
PRICES_PER_MTOK = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0),
}
BATCH_DISCOUNT = 0.5
# anthropic SDK 1.x dropped the temperature kwarg; models that still accept it get it via extra_body.
# Sonnet 5.5 rejects non-default sampling values, so it runs at its default.
MODELS_ACCEPTING_TEMPERATURE = frozenset({"claude-haiku-4-5"})
