import sys
from typing import List

from backend import (
    ConversationLogger,
    apply_system_prompt_update,
    build_leak_registry,
    build_system_prompt,
    classify_sensitive_content_cached,
    create_chat,
    likely_leak,
    run_attack_suite,
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
    current_user_context = user_context
    current_system_prompt = build_system_prompt(current_user_context)
    leak_registry = build_leak_registry(current_system_prompt, current_user_context)
    client, chat = create_chat(api_key, current_system_prompt)
    messages: List[tuple[str, str, bool, bool]] = []
    user_history: List[str] = []

    template = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prompt Injection Test</title>
  <style>
    body { font-family: sans-serif; margin: 24px; max-width: 900px; }
    .msg { border: 1px solid #ccc; border-radius: 8px; padding: 10px; margin: 8px 0; }
    .user { background: #f5faff; }
    .assistant { background: #f8f8f8; }
    .security { background: #fffcf0; border-color: #fbc02d; }
    .leak { border-color: #d32f2f; }
    textarea { width: 100%; min-height: 80px; }
    button { margin-right: 8px; }
    .actions { display: flex; gap: 8px; align-items: center; margin-top: 10px; }
    .actions form { margin: 0; }
  </style>
</head>
<body>
  <h1>Prompt Injection Test</h1>
  <p>セッションID: {{ session_id }}</p>
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
      <strong>
        {% if role == 'security' %}🛡️ Security Guard
        {% elif role == 'assistant' %}🤖 AI Assistant
        {% elif role == 'user' %}👤 User
        {% else %}{{ role }}{% endif %}
      </strong>
      {% if leaked and show_flag %} [LEAK DETECTED]{% endif %}<br>
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
            user_context=current_user_context,
        )

    @app.post("/user-context")
    def update_user_context():
        nonlocal client, chat, current_user_context, current_system_prompt, leak_registry
        updated = (request.form.get("user_context") or "").strip()
        if not updated:
            return redirect(url_for("index"))

        try:
            (
                current_user_context,
                current_system_prompt,
                leak_registry,
                client,
                chat,
            ) = apply_system_prompt_update(
                api_key=api_key,
                user_context=updated,
                logger=logger,
            )
        except ValueError:
            return redirect(url_for("index"))

        user_history.clear()
        messages.append(("security", "[INFO] User Context を更新し、新しい会話セッションを開始しました。", False, False))
        return redirect(url_for("index"))

    @app.post("/send")
    def send():
        nonlocal client, chat
        text = (request.form.get("message") or "").strip()
        if not text:
            return redirect(url_for("index"))

        input_class = classify_sensitive_content_cached(text=text, client=client)
        if input_class.is_sensitive:
            messages.append(
                (
                    "security",
                    f"[INPUT RISK:{input_class.risk}] {', '.join(input_class.categories) or input_class.rationale}",
                    False,
                    False,
                )
            )

        logger.write(mode="chat", role="user", message=text)
        try:
            resp = chat.send_message(text)
        except RuntimeError as e:
            if "client has been closed" not in str(e):
                raise
            client, chat = create_chat(api_key, current_system_prompt)
            for prev in user_history:
                chat.send_message(prev)
            resp = chat.send_message(text)

        answer = resp.text or ""
        leak_result = likely_leak(answer, leak_registry, client=client)
        leaked = leak_result.leaked
        logger.write(mode="chat", role="assistant", message=answer, leak_detected=leaked)
        user_history.append(text)
        messages.append(("user", text, False, False))
        flag = "リーク" if leaked else "安全"
        if leaked:
            messages.append(("security", f"[LEAK DETECTED] セキュリティ保護のため、AIの回答を遮断しました。", True, False))
        
        answer = f"[{flag}] {answer}"
        messages.append(("assistant", answer, leaked, True))
        return redirect(url_for("index"))

    @app.post("/attack")
    def attack():
        nonlocal client, chat
        try:
            results = run_attack_suite(chat, leak_registry, logger=logger, client=client)
        except RuntimeError as e:
            if "client has been closed" not in str(e):
                raise
            client, chat = create_chat(api_key, current_system_prompt)
            for prev in user_history:
                chat.send_message(prev)
            results = run_attack_suite(chat, leak_registry, logger=logger, client=client)

        for prompt, reply, leaked in results:
            status = "リーク" if leaked else "安全"
            user_history.append(prompt)
            messages.append(("user", f"[ATTACK] {prompt}", False, False))
            messages.append(("assistant", f"[{status}] {reply}", leaked, True))
        return redirect(url_for("index"))

    @app.post("/reset")
    def reset():
        nonlocal client, chat
        client, chat = create_chat(api_key, current_system_prompt)
        messages.clear()
        user_history.clear()
        return redirect(url_for("index"))

    print(f"Web UI 起動: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
    return 0
