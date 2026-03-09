import sys
from typing import List

from backend import (
    ConversationLogger,
    apply_system_prompt_update,
    build_leak_registry,
    build_system_prompt,
    create_chat,
    generate_canary_token,
    likely_leak,
    run_attack_suite,
    generate_refusal_response,
    classify_sensitive_content_cached,
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
    current_system_prompt = build_system_prompt(current_user_context, canary_token=canary_token)
    leak_registry = build_leak_registry(canary_token)
    
    client, chat = create_chat(api_key, current_system_prompt)
    messages: List[tuple[str, str, bool, bool]] = []

    template = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prompt Injection (Few-Shot)</title>
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
  <h1>Prompt Injection (Few-Shot)</h1>
  <p>セッションID: {{ session_id }} | カナリートークン: {{ canary_token }}</p>
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
      <div style="white-space: pre-wrap; margin: 6px 0 0 0;">{{ text }}</div>
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
                canary_token=canary_token,
                logger=logger
            )
        except ValueError:
            return redirect(url_for("index"))

        messages.append(("security", "[INFO] User Context を更新し、新しい会話セッションを開始しました。", False, False))
        return redirect(url_for("index"))

    @app.post("/send")
    def send():
        nonlocal client, chat
        text = (request.form.get("message") or "").strip()
        if not text:
            return redirect(url_for("index"))

        messages.append(("user", text, False, False))
        logger.write("user", text)

        # 入力ガード (Input Guard) の追加
        input_class = classify_sensitive_content_cached(text=text, client=client)
        if input_class.is_sensitive:
            security_msg = f"[INPUT RISK:{input_class.risk}] {', '.join(input_class.categories) or input_class.rationale}"
            # リーク検知時と同じ「安全なお断り」を生成
            block_msg = generate_refusal_response(api_key)
            
            messages.append(("security", security_msg, False, False))
            messages.append(("assistant", f"[遮断] {block_msg}", False, False))
            logger.write("security", security_msg)
            logger.write("assistant", block_msg)
            return redirect(url_for("index"))
        
        try:
            resp = chat.send_message(text)
            answer = resp.text or ""
        except Exception as e:
            answer = f"[ERROR] {str(e)}"

        leak_res = likely_leak(answer, leak_registry)
        leaked = leak_res.leaked
        
        if leaked:
            messages.append(("security", f"[LEAK DETECTED] セキュリティ保護のため、AIの回答を遮断しました。", True, False))
            # リーク検知時は安全なお断りメッセージを生成して差し替える
            answer = generate_refusal_response(api_key)
        
        status = "遮断" if leaked else "安全"
        answer_display = f"[{status}] {answer}"
        messages.append(("assistant", answer_display, leaked, True))
        logger.write("assistant", answer_display, leak_detected=leaked)
        return redirect(url_for("index"))

    @app.post("/attack")
    def attack():
        nonlocal client, chat
        results = run_attack_suite(api_key, current_system_prompt, leak_registry)
        
        for prompt, reply, leaked in results:
            status = "遮断" if leaked else "安全"
            messages.append(("user", f"[ATTACK] {prompt}", False, False))
            logger.write("attack_user", prompt)
            
            if leaked:
                messages.append(("security", f"[LEAK DETECTED] 攻撃により情報が漏洩しました。", True, False))
            
            messages.append(("assistant", f"[{status}] {reply}", leaked, True))
            logger.write("attack_assistant", reply, leak_detected=leaked)
            
        return redirect(url_for("index"))

    @app.post("/reset")
    def reset():
        nonlocal client, chat
        client, chat = create_chat(api_key, current_system_prompt)
        messages.clear()
        return redirect(url_for("index"))

    @app.get("/reset")
    def reset_get():
        return reset()

    print(f"Web UI 起動: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
    return 0
