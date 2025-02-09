import os
from src.LoggerFactory import LoggerFactory

METACULUS_API_TOKEN = os.getenv("METACULUS_API_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_CLIENT_SECRET = os.getenv("ASKNEWS_CLIENT_SECRET")
METACULUS_OPENAI_PROXY_URL = os.getenv("METACULUS_OPENAI_PROXY_URL")


if METACULUS_TOKEN is None:
    raise ValueError("The environment variable METACULUS_TOKEN is not set.")


OPENAI_MODEL_SMART = os.getenv("OPENAI_MODEL")
LLM_TO_USE = os.getenv("LLM_TO_USE")
LLM_MODEL_CONFIG = os.getenv("LLM_MODEL_CONFIG")
TEXT_EMBEDDING_MODEL = "text-embedding-3-small" # TODO: Hacer ENV VAR

POST_PREDICTIONS = os.getenv("POST_PREDICTIONS")


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_TO_CONSOLE = os.getenv("LOG_TO_CONSOLE", True)
LOGS_FILE_DIR = os.getenv("LOGS_FILE_DIR", "logs")
LOGS_FILE_NAME = os.getenv("LOGS_FILE_NAME", "tmp.log")

logger_factory = LoggerFactory(
    log_level=LOG_LEVEL,
    log_to_console=LOG_TO_CONSOLE,
    logs_file_dir=LOGS_FILE_DIR,
    logs_file_name=LOGS_FILE_NAME)

from src.openai_utils import make_proxied_ChatOpenAI_LLM
llm_smart = make_proxied_ChatOpenAI_LLM(temperature=0.1)


AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api2"
WARMUP_TOURNAMENT_ID = 3294
ACTUAL_TOURNAMENT_ID = 3349
WARMUP_TOURNAMENT_ID = ACTUAL_TOURNAMENT_ID
BOT_TOURNAMENT_IDS = [3294, 3349] # TODO: Hacer ENV VAR
