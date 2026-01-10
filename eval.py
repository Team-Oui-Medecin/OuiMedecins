from inspect_ai import eval
from inspect_evals.sycophancy import sycophancy

# ---- Define the vLLM server here ----
MODEL_BASE_URL = "http://130.61.138.189:8000/v1"
MODEL_NAME = "openai/openai/gpt-oss-20b"
API_KEY = "EMPTY"

# ---- Run eval ----
LIMIT = 100
TIME_LIMIT = 60  # abort sample if it takes more than TIME_LIMIT seconds
MAX_CONNECTIONS = 16 # number of parallel connections to vLLM server (don't go above 16)
IGNORE_ERRORS = True
FAIL_ON_ERROR = False
eval(sycophancy, model=MODEL_NAME, model_base_url=MODEL_BASE_URL, limit=LIMIT, time_limit=TIME_LIMIT, max_connections=MAX_CONNECTIONS, ignore_errors=IGNORE_ERRORS, fail_on_error=FAIL_ON_ERROR)