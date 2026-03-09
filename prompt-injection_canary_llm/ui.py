import sys
from typing import List

from backend import (
    ConversationLogger,
    DualChatSession,
    apply_system_prompt_update,
    check_canary_leak,
    generate_canary_token,
    run_attack_suite,
    generate_refusal_response,
)


def run_web_app(
    api_key: str,
    user_context: str,
    logger: ConversationLogger,
    host: str,
    port: int,
) -> int:
    try:
        from flask import Flask, redirect, render_template_string, request, url_for
    except ImportError:
        print("Webモードには Flask が必要です。`pip install flask` を実行してください。", file=sys.stderr)
        return 1

    app = Flask(__name__)
    canary_token = generate_canary_token()
    current_user_context = user_context
    session = DualChatSession(api_key, current_user_context, canary_token, logger=logger)
    messages: List[tuple[str, str, bool, bool]] = []

    template = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prompt Injection (カナリートークン＆二重LLM構成)</title>
  <style>
    body { font-family: sans-serif; margin: 24px; max-width: 900px; }
    .msg { border: 1px solid #ccc; border-radius: 8px; padding: 10px; margin: 8px 0; }
    .user { background: #f5faff; }
    .assistant { background: #f8f8f8; }
    .leak { border-color: #d32f2f; }
    textarea { width: 100%; min-height: 80px; }
    button { margin-right: 8px; }
    .actions { display: flex; gap: 8px; align-items: center; margin-top: 10px; }
    .actions form { margin: 0; }
  </style>
</head>
<body>
  <h1>Prompt Injection (カナリートークン＆二重LLM構成)</h1>
  <p>セッションID: {{ session_id }} | カナリートークン: <code>{{ canary_token }}</code></p>
  <form method="post" action="{{ url_for('update_user_context') }}">
    <label for="user_context"><strong>User Context</strong></label><br>
    <textarea id="user_context" name="user_context" placeholder="ユーザ情報を入力">{{ user_context }}</textarea><br>
    <button type="submit">User Context 適用</button>
  </form>
  <hr>
  <form id="send_form" method="post" action="{{ url_for('send') }}">
    <textarea id="message_input" name="message" placeholder="メッセージを入力"></textarea><br>
  </form>
  <div class="actions">
    <button type="submit" form="send_form">送信</button>
    <form method="post" action="{{ url_for('attack') }}">
      <button type="submit">Attack Suite 実行</button>
    </form>
    <form method="post" action="{{ url_for('reset') }}">
      <button type="submit">会話リセット</button>
    </form>
  </div>
  <hr>
  {% for role, text, leaked, show_flag in messages %}
    <div class="msg {{ role }} {% if leaked %}leak{% endif %}">
      <strong>{{ role }}</strong>{% if leaked and show_flag %} [LEAK DETECTED]{% endif %}<br>
      <pre style="white-space: pre-wrap; margin: 6px 0 0 0;">{{ text }}</pre>
    </div>
  {% endfor %}
</body>
</html>
""".strip()

    @app.get("/")
    def index():
        return render_template_string(
            template,
            messages=messages,
            session_id=logger.session_id,
            canary_token=canary_token,
            user_context=current_user_context,
        )

    @app.post("/user-context")
    def update_user_context():
        nonlocal session, current_user_context
        updated = (request.form.get("user_context") or "").strip()
        if not updated:
            return redirect(url_for("index"))

        try:
            (
                current_user_context,
                session,
            ) = apply_system_prompt_update(
                api_key=api_key,
                user_context=updated,
                canary_token=canary_token,
                logger=logger,
            )
        except ValueError:
            return redirect(url_for("index"))
 
        messages.append(("assistant", "[INFO] User Context を更新し、新しい会話セッションを開始しました。", False, False))
        return redirect(url_for("index"))

    @app.post("/send")
    def send():
        nonlocal session
        text = (request.form.get("message") or "").strip()
        if not text:
            return redirect(url_for("index"))

        input_text = text
        logger.write(mode="chat", role="user", message=input_text)
        
        answer = session.send_message(input_text)
        leaked = check_canary_leak(answer, canary_token)
        
        if leaked:
            # リーク時は安全なお断りメッセージを生成して差し替える
            answer = generate_refusal_response(api_key)
            
        logger.write(mode="chat", role="assistant", message=answer, leak_detected=leaked)
        messages.append(("user", text, False, False))
        
        status = "遮断" if leaked else "安全"
        display_answer = f"[{status}] {answer}"
        messages.append(("assistant", display_answer, leaked, True))
        return redirect(url_for("index"))

    @app.post("/attack")
    def attack():
        nonlocal session
        try:
            results = run_attack_suite(session, logger=logger)
        except RuntimeError as e:
            if "client has been closed" not in str(e):
                raise
            session = DualChatSession(api_key, current_user_context, canary_token, logger=logger)
            results = run_attack_suite(session, logger=logger)

        for res in results:
            leaked = check_canary_leak(res.reply, canary_token)
            
            reply_text = res.reply
            if leaked:
                reply_text = generate_refusal_response(api_key)
                
            status = "遮断" if leaked else "安全"
            messages.append(("user", f"[ATTACK] {res.prompt}", False, False))
            messages.append(("assistant", f"[{status}] {reply_text}", leaked, True))
        return redirect(url_for("index"))

    @app.post("/reset")
    def reset():
        nonlocal session, canary_token
        canary_token = generate_canary_token()
        session = DualChatSession(api_key, current_user_context, canary_token, logger=logger)
        messages.clear()
        return redirect(url_for("index"))

    print(f"Web UI 起動: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
    return 0
