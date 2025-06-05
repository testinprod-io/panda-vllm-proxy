import logging, os, datetime
from logging.config import dictConfig

# logfmt encoder
def lf_encode(d: dict) -> str:
    def esc(v: str) -> str:
        s = str(v)
        if any(ch in s for ch in (' ', '"', '=')):
            s = s.replace('"', r'\"')
            return f'"{s}"'
        return s
    return ' '.join(f'{k}={esc(v)}' for k, v in d.items())

STD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname",
    "filename", "module", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "processName", "process",
}

class LeanLogfmt(logging.Formatter):
    CORE = ("ts", "level", "logger", "msg")

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": datetime.datetime.utcfromtimestamp(record.created)
                      .isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Copy ONLY user-supplied extras
        for k, v in record.__dict__.items():
            # Drop STD_ATTRS
            if k not in STD_ATTRS and not k.startswith("_"):
                base[k] = v

        return lf_encode(base)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"lean": {"()": LeanLogfmt}},
    "handlers": {"console": {"class": "logging.StreamHandler",
                             "formatter": "lean"}},
    "root": {"level": LOG_LEVEL, "handlers": ["console"]},
    "loggers": {
        "uvicorn.access": {"level": "INFO", "handlers": ["console"],
                           "propagate": False},
        "uvicorn.error":  {"level": "INFO", "handlers": ["console"],
                           "propagate": False},
        "app":            {"level": LOG_LEVEL, "handlers": ["console"],
                           "propagate": False},
    },
})

log = logging.getLogger("app")