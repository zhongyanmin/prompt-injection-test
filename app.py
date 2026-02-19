import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4

from backend import (
    ConversationLogger,
    run_cli,
)
from ui import run_web_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Prompt Injection Test Chat (Gemini)")
    parser.add_argument("--log-file", default="logs/chat_log.log")
    parser.add_argument("--web", action="store_true", help="FlaskのWeb UIで起動")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY が未設定です。", file=sys.stderr)
        return 1

    user_context = ""
    session_id = uuid4().hex[:12]
    logger = ConversationLogger(path=Path(args.log_file), session_id=session_id)

    if args.web:
        return run_web_app(
            api_key=api_key,
            user_context=user_context,
            logger=logger,
            host=args.host,
            port=args.port,
        )

    return run_cli(
        api_key=api_key,
        user_context=user_context,
        logger=logger,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
