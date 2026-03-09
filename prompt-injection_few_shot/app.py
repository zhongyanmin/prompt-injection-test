import argparse
import os
import sys

def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Prompt Injection (Few-Shot)")
    parser.add_argument("--log-file", default="logs/chat_log.log")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY が未設定です。", file=sys.stderr)
        return 1

    from uuid import uuid4
    from pathlib import Path
    from backend import ConversationLogger
    from ui import run_web_app

    user_context = ""
    session_id = uuid4().hex[:12]
    logger = ConversationLogger(path=Path(args.log_file), session_id=session_id)

    return run_web_app(
        api_key=api_key,
        user_context=user_context,
        logger=logger, 
        host=args.host,
        port=args.port,
    )

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
