"""Nexora AI — Python 3.10+. Install Ollama, run ollama pull gemma3:4b, then py Nexora_AI_Pulsuz.py"""
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import threading
import time
from urllib.parse import urlencode
import webbrowser

DATA = Path.home() / ".nexora-ai"
MODEL = os.environ.get("NEXORA_LOCAL_MODEL", "gemma3:4b")
LANG = {"az": "Azerbaijani", "tr": "Turkish", "en": "English"}
IMAGE = re.compile(r"data:image/(?:jpeg|png|webp);base64,[A-Za-z0-9+/]+={0,2}")

COMFY = "http://127.0.0.1:8188"
CHECKPOINT = "sd_xl_turbo_1.0_fp16.safetensors"


def database():
    connection = sqlite3.connect(DATA / "nexora.sqlite3")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def initialize():
    DATA.mkdir(mode=0o700, parents=True, exist_ok=True)
    with database() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
                salt BLOB NOT NULL, password BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
                expires INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
                title TEXT NOT NULL, messages TEXT NOT NULL
            );
        """)
        db.execute("DELETE FROM sessions WHERE expires <= CAST(strftime('%s','now') AS INTEGER)")


def password_hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=salt, n=32768, r=8, p=1,
                          maxmem=64 * 1024 * 1024)


def comfy_call(path, payload=None, image=False):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(COMFY + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=15) as response:
            if image:
                binary = response.read(8_000_001)
                if len(binary) > 8_000_000 or not binary.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ApiError(502, "ComfyUI etibarlı PNG qaytarmadı.")
                return binary
            return json.load(response)
    except HTTPError as error:
        raise ApiError(502, f"ComfyUI xətası ({error.code}). Model faylını və ComfyUI pəncərəsini yoxla.") from None
    except (URLError, TimeoutError):
        raise ApiError(503, "Şəkil yaratmaq üçün ComfyUI-ni aç (127.0.0.1:8188).") from None


def create_picture(prompt):
    checkpoints = comfy_call("/models/checkpoints")
    if CHECKPOINT not in checkpoints:
        raise ApiError(503, f"{CHECKPOINT} faylını ComfyUI/models/checkpoints qovluğuna yerləşdir.")
    nodes = {
        "1": ("CheckpointLoaderSimple", {"ckpt_name": CHECKPOINT}),
        "2": ("CLIPTextEncode", {"clip": ["1", 1], "text": prompt}),
        "3": ("CLIPTextEncode", {"clip": ["1", 1], "text": ""}),
        "4": ("EmptyLatentImage", {"width": 512, "height": 512, "batch_size": 1}),
        "5": ("KSampler", {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                          "latent_image": ["4", 0], "seed": secrets.randbits(48), "steps": 4,
                          "cfg": 1.0, "sampler_name": "euler_ancestral", "scheduler": "normal",
                          "denoise": 1.0}),
        "6": ("VAEDecode", {"samples": ["5", 0], "vae": ["1", 2]}),
        "7": ("PreviewImage", {"images": ["6", 0]}),
    }
    workflow = {key: {"class_type": name, "inputs": inputs}
                for key, (name, inputs) in nodes.items()}
    job = comfy_call("/prompt", {"prompt": workflow})
    job_id = job.get("prompt_id", "")
    if not re.fullmatch(r"[a-f0-9-]{36}", job_id):
        raise ApiError(502, "ComfyUI şəkil tapşırığını qəbul etmədi.")
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        result = comfy_call("/history/" + job_id).get(job_id)
        if result:
            images = result.get("outputs", {}).get("7", {}).get("images", [])
            if not images:
                raise ApiError(502, "Şəkil yaranmadı. ComfyUI pəncərəsində xətaya bax.")
            output = images[0]
            query = urlencode({key: output[key] for key in ("filename", "subfolder", "type")})
            binary = comfy_call("/view?" + query, image=True)
            return {"role": "assistant", "image": "data:image/png;base64," + base64.b64encode(binary).decode("ascii")}
        time.sleep(2)
    raise ApiError(504, "Şəkil vaxtında hazır olmadı. ComfyUI-də tapşırığın vəziyyətini yoxla.")


def generate(messages, prompt, image, mode, language):
    if mode == "image":
        return create_picture(prompt)
    history = [{"role": entry["role"], "content": entry["text"]}
               for entry in messages[-12:] if entry.get("text")]
    user_message = {"role": "user", "content": prompt or "Describe and analyze this image."}
    if image:
        user_message["images"] = [image.split(",", 1)[1]]
    instruction = ("You are Nexora AI, an assistant for coding, mathematics, history "
                   "and general knowledge. Answer by default in " + LANG[language] +
                   ". Explain important steps and be clear about uncertainty. "
                   "Use fenced code blocks with language names.")
    instruction += (" Ignore language errors in earlier assistant messages. "
                    "Never claim to remember details outside this conversation. "
                    "For image creation, tell the user to select the image mode in Nexora.")
    if language == "az":
        instruction += (" Azərbaycan dilində təbii, qısa və aydın cavab ver. "
                        "Ədəbi Azərbaycan dilinin orfoqrafiyasına əməl et. "
                        "Ə, ğ, ı, i, ö, ş, ü, ç hərflərini düzgün yaz. "
                        "İstifadəçi diakritikasız yazsa da, düzgün Azərbaycan dilində cavab ver. "
                        "Söz uydurma, mövzuya aid olmayan ifadələr işlətmə. "
                        "Cavabı göndərməzdən əvvəl yazılışı və mənanı yoxla. "
                        "Məsələn, 'Necəsən?' sualına 'Yaxşıyam, sağ olun! Siz necəsiniz?' yaz.")
    if mode == "code":
        instruction += " Write maintainable, runnable code and explain how to use it."
    payload = {"model": MODEL, "stream": False, "options": {"temperature": 0.2},
               "messages": [{"role": "system", "content": instruction}, *history, user_message]}
    request = Request("http://127.0.0.1:11434/api/chat", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=360) as response:
            answer = json.load(response).get("message", {}).get("content", "").strip()
    except HTTPError as error:
        with error:
            if error.code == 404:
                raise ApiError(503, "Modeli endirmək üçün CMD-də yaz: ollama pull " + MODEL) from None
            detail = json.loads(error.read()).get("error", "")
        raise ApiError(502, "Ollama: " + str(detail or error.code)) from None
    except (URLError, TimeoutError):
        raise ApiError(503, "Ollama cavab vermədi. Ollama proqramını aç və yenidən yoxla.") from None
    if not answer:
        raise ApiError(502, "Ollama boş cavab qaytardı.")
    return {"role": "assistant", "text": answer}


class ApiError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


class Handler(BaseHTTPRequestHandler):
    def send(self, status, data, cookie=None):
        html = isinstance(data, str)
        body = data.encode("utf-8") if html else json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", ("text/html" if html else "application/json") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                         "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                         "frame-ancestors 'none'; object-src 'none'; base-uri 'none'")
        if cookie is not None:
            self.send_header("Set-Cookie", "nexora=" + (cookie or "; Max-Age=0") + "; Path=/; HttpOnly; SameSite=Strict")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.run_request(False)

    def do_POST(self):
        self.run_request(True)

    def run_request(self, post):
        host = "127.0.0.1:" + str(self.server.server_port)
        try:
            if self.headers.get("Host") != host:
                raise ApiError(403, "Invalid host.")
            if post and (self.headers.get("Origin") != "http://" + host or
                         self.headers.get_content_type() != "application/json"):
                raise ApiError(403, "Invalid origin.")
            if not post and self.path == "/":
                self.send(200, PAGE)
                return
            data = {}
            if post:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 9_000_000:
                    raise ApiError(413, "Request too large.")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ApiError(400, "Invalid request.")
            token_match = re.search(r"(?:^|;\s*)nexora=([a-f0-9]{64})(?:;|$)",
                                    self.headers.get("Cookie", ""))
            token = token_match[1] if token_match else ""
            with database() as db:
                result, cookie = self.route(db, token, data, post)
            self.send(200, result, cookie)
        except ApiError as error:
            self.send(error.status, {"error": error.message})
        except (ValueError, TypeError):
            self.send(400, {"error": "Invalid request."})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            self.log_error("Request failed unexpectedly")
            self.send(500, {"error": "Nexora could not complete this request."})

    def route(self, db, token, data, post):
        path = self.path
        if post and path in ("/api/login", "/api/register"):
            name, password = data.get("name", ""), data.get("password", "")
            if not isinstance(name, str) or not isinstance(password, str):
                raise ApiError(400, "Invalid credentials.")
            name = name.strip().casefold()
            if not 2 <= len(name) <= 60 or not 8 <= len(password) <= 200:
                raise ApiError(400, "Username: 2–60 characters. Password: 8–200 characters.")
            user = db.execute("SELECT * FROM users WHERE name=?", (name,)).fetchone()
            if path.endswith("/register"):
                salt = secrets.token_bytes(16)
                try:
                    user_id = db.execute("INSERT INTO users(name,salt,password) VALUES(?,?,?)",
                                         (name, salt, password_hash(password, salt))).lastrowid
                except sqlite3.IntegrityError:
                    raise ApiError(409, "This username already exists.") from None
            else:
                digest = password_hash(password, user["salt"] if user else bytes(16))
                if not user or not hmac.compare_digest(digest, user["password"]):
                    raise ApiError(401, "Incorrect username or password.")
                user_id = user["id"]
            db.execute("DELETE FROM sessions WHERE token=?", (token,))
            token = secrets.token_hex(32)
            db.execute("INSERT INTO sessions VALUES(?,?,CAST(strftime('%s','now') AS INTEGER)+86400)", (token, user_id))
            return {"name": name}, token

        user = db.execute("SELECT users.id, name FROM sessions JOIN users ON users.id=user_id "
                          "WHERE token=? AND expires>CAST(strftime('%s','now') AS INTEGER)", (token,)).fetchone()
        if not post and path == "/api/me":
            return {"name": user["name"] if user else ""}, None
        if not user:
            raise ApiError(401, "Please sign in.")
        user_id = user["id"]
        if post and path == "/api/logout":
            db.execute("DELETE FROM sessions WHERE token=?", (token,))
            return {}, ""
        if not post and path == "/api/chats":
            rows = db.execute("SELECT id,title FROM chats WHERE user_id=? ORDER BY id DESC", (user_id,))
            return [dict(row) for row in rows], None
        if not post and re.fullmatch(r"/api/chats/[0-9]+", path):
            row = self.chat(db, int(path.rsplit("/", 1)[1]), user_id)
            return json.loads(row["messages"]), None
        if post and path == "/api/send":
            return self.chat_send(db, user_id, data), None
        raise ApiError(404, "Page not found.")

    def chat(self, db, chat_id, user_id):
        row = db.execute("SELECT * FROM chats WHERE id=? AND user_id=?", (chat_id, user_id)).fetchone()
        if not row:
            raise ApiError(404, "Chat not found.")
        return row

    def chat_send(self, db, user_id, data):
        prompt, image = data.get("text", ""), data.get("image", "")
        mode, language, chat_id = data.get("mode"), data.get("language"), data.get("id")
        if not isinstance(prompt, str) or len(prompt) > 16000 or not isinstance(image, str):
            raise ApiError(400, "Invalid message.")
        if not isinstance(language, str) or language not in LANG or mode not in ("chat", "code", "image"):
            raise ApiError(400, "Invalid mode or language.")
        prompt = prompt.strip()
        if image and (len(image) > 8_000_000 or not IMAGE.fullmatch(image)):
            raise ApiError(400, "Use a JPEG, PNG or WebP image up to 5 MB.")
        if mode == "image" and (not prompt or image):
            raise ApiError(400, "Yaradılacaq şəkli mətnlə təsvir et.")
        if not prompt and not image:
            raise ApiError(400, "Enter a message or attach an image.")
        if chat_id is not None and type(chat_id) is not int:
            raise ApiError(400, "Invalid chat ID.")
        previous = self.chat(db, chat_id, user_id)["messages"] if chat_id is not None else "[]"
        messages = json.loads(previous)
        answer = generate(messages, prompt, image, mode, language)
        messages.extend([{"role": "user", "text": prompt, "image": image}, answer])
        saved = json.dumps(messages, ensure_ascii=False)
        if chat_id is None:
            chat_id = db.execute("INSERT INTO chats(user_id,title,messages) VALUES(?,?,?)",
                                 (user_id, prompt[:65] or "📷", saved)).lastrowid
        else:
            updated = db.execute("UPDATE chats SET messages=? WHERE id=? AND user_id=? AND messages=?",
                                 (saved, chat_id, user_id, previous))
            if not updated.rowcount:
                raise ApiError(409, "This chat changed in another tab. Reopen it and retry.")
        return {"id": chat_id, "messages": messages}


PAGE = r"""<!doctype html>
<html lang="az"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nexora AI</title>
<style>
:root{color-scheme:dark;--bg:#100e18;--panel:#191625;--card:#231e33;--line:#373049;--ink:#f5f0ff;--muted:#bcb1cc;--accent:#b89aff;--button:#dbcaff}
:root[data-theme=light]{color-scheme:light;--bg:#f9f6ff;--panel:#eee7fa;--card:#fff;--line:#d6cbe8;--ink:#241639;--muted:#665575;--accent:#6337a8;--button:#dbcaff}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
button,select,input,textarea{font:inherit;color:inherit}
button,select,input{border:1px solid var(--line);border-radius:10px;background:var(--card);padding:9px 13px}
button{cursor:pointer}button:hover{border-color:var(--accent)}button:disabled{opacity:.5;cursor:default}
.primary{background:var(--button);color:#28174a;border-color:transparent;font-weight:700}
#app{min-height:100dvh;display:grid;grid-template-columns:245px minmax(0,1fr)}
aside{padding:23px 16px;background:var(--panel);border-right:1px solid var(--line);min-width:0}
.brand{display:flex;align-items:center;gap:11px;font-size:19px;font-weight:800;margin:0 0 24px}
.mark{display:grid;place-items:center;width:36px;height:36px;border-radius:12px;background:linear-gradient(140deg,#e2b4ff,#8066e6);color:#211437;font-size:22px}
aside button{width:100%;text-align:left;margin:0 0 9px}#history{max-height:50vh;overflow:auto}#history button{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.small{font-size:12px;color:var(--muted)}
main{min-width:0;max-width:1000px;width:100%;margin:0 auto;padding:22px clamp(14px,4vw,48px);display:flex;flex-direction:column}
header,.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap}header{justify-content:space-between}
#landing{text-align:center;margin:auto;padding:55px 0}#landing .mark{margin:0 auto;width:66px;height:66px;font-size:37px;border-radius:21px;box-shadow:0 0 60px #9866e65b}
h1{font-size:clamp(32px,6vw,60px);letter-spacing:-.07em;line-height:1.05;margin:20px 0 12px}
#landing p{color:var(--muted);margin:0 0 24px}#examples{display:flex;flex-wrap:wrap;justify-content:center;gap:8px}
#messages{min-width:0;overflow-wrap:anywhere}article{padding:16px;margin:18px 0;border:1px solid var(--line);border-radius:16px}article.user{background:var(--panel);margin-left:clamp(10px,7vw,100px)}
article p{white-space:pre-wrap;margin:10px 0}article img{display:block;max-width:min(100%,520px);border-radius:13px;margin:10px 0}pre{overflow:auto;background:var(--card);padding:14px;border-radius:9px;white-space:pre}a{color:var(--accent)}
form#composer{position:sticky;bottom:8px;background:var(--card);border:1px solid var(--line);border-radius:20px;padding:12px;box-shadow:0 6px 26px #0003;margin-top:auto}
textarea{display:block;width:100%;resize:vertical;background:none;border:0;padding:8px 6px;min-height:66px;outline-color:var(--accent)}.spacer{flex:1}
dialog{color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:23px;width:min(440px,92vw)}dialog::backdrop{background:#0b0813b8}dialog label{display:grid;gap:5px;margin:13px 0}dialog h2{margin-top:0}video{width:100%;border-radius:10px}
#status{margin:6px 4px;color:var(--muted)}#attachment{max-width:85px;border-radius:8px}#authError{color:var(--accent)}
@media(max-width:650px){#app{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid var(--line);padding:12px}.brand{margin-bottom:10px}aside button{width:auto}#history{display:flex;gap:6px;max-width:100%}#history button{max-width:145px;flex-shrink:0}}
article{width:fit-content;min-width:min(135px,100%);max-width:90%}
article.user{margin-left:auto;background:var(--card);border-color:var(--accent)}
article.assistant{margin-right:auto;background:var(--panel)}
.typing{display:flex;gap:6px;margin:12px 0 4px}
.typing span{width:7px;height:7px;border-radius:50%;background:var(--accent);animation:pulse 1.2s infinite ease-in-out}
.typing span:nth-child(2){animation-delay:.15s}.typing span:nth-child(3){animation-delay:.3s}
@keyframes pulse{0%,80%,100%{opacity:.4}40%{opacity:1;transform:translateY(-3px)}}
@media(prefers-reduced-motion:reduce){.typing span{animation:none}}
</style></head>
<body><div id="app">
<aside><div class="brand"><span class="mark">✦</span>Nexora AI</div><button id="newChat" data-label="newChat"></button><p class="small" data-label="history"></p><div id="history"></div></aside>
<main><header><button id="account"></button><button id="settingsButton" data-label="settings"></button></header>
<section id="landing"><div class="mark">✦</div><h1 data-label="hero"></h1><p data-label="intro"></p><div id="examples"></div></section>
<div id="messages" aria-live="polite"></div>
<form id="composer"><textarea id="prompt" maxlength="16000" rows="2"></textarea>
<div id="preview" class="row" hidden><img id="attachment" alt=""><button id="remove" type="button" data-label="remove"></button></div>
<div class="row"><select id="mode"></select><button id="upload" type="button" data-label="upload"></button><input id="file" type="file" accept="image/png,image/jpeg,image/webp" hidden><button id="cameraButton" type="button" data-label="camera"></button><span class="spacer"></span><button id="send" class="primary" data-label="send"></button></div>
<div id="status" role="status" class="small"></div></form></main></div>
<dialog id="auth"><form id="authForm"><h2 data-label="accountTitle"></h2>
<label><span data-label="name"></span><input id="name" required minlength="2" maxlength="60" autocomplete="username"></label>
<label><span data-label="password"></span><input id="password" type="password" required minlength="8" maxlength="200" autocomplete="current-password"></label>
<p id="authError" role="alert"></p>
<div class="row"><button id="login" class="primary" data-label="login" value="login"></button><button id="register" data-label="register" value="register"></button><button type="button" data-close="auth" data-label="close"></button></div></form></dialog>
<dialog id="settings"><h2 data-label="settings"></h2><label><span data-label="language"></span><select id="language"><option value="az">Azərbaycanca</option><option value="tr">Türkçe</option><option value="en">English</option></select></label>
<label><span data-label="theme"></span><select id="theme"></select></label><div class="row"><button id="logout" data-label="logout" hidden></button><button data-close="settings" data-label="close"></button></div></dialog>
<dialog id="camera"><h2 data-label="camera"></h2><video id="video" playsinline autoplay muted></video><div class="row"><button id="snap" class="primary" data-label="snap" disabled></button><button data-close="camera" data-label="close"></button></div></dialog>
<script>
const by = id => document.getElementById(id);
const words = {
 az:{newChat:"＋ Yeni söhbət",history:"Söhbət tarixçəsi",settings:"⚙ Ayarlar",hero:"Bu gün nə edək?",intro:"Məsələ həll et, kod yaz, tarixi araşdır və ya yüklədiyin şəkli izah etdir.",remove:"Şəkli sil",upload:"↑ Şəkil yüklə",camera:"Kamera",send:"Göndər ↑",accountTitle:"Nexora hesabı",name:"İstifadəçi adı",password:"Parol (ən azı 8 simvol)",login:"Daxil ol",register:"Qeydiyyat",close:"Bağla",language:"Dil",theme:"Görünüş",logout:"Çıxış",chat:"Söhbət",code:"Kod",image:"Şəkil yarat",dark:"Qaranlıq",light:"Açıq",prompt:"Nexora AI-yə yaz…",imagePrompt:"Yaradılacaq şəkli təsvir et…",wait:"Nexora cavab hazırlayır…",badImage:"5 MB-a qədər JPG, PNG və ya WebP seç.",cameraError:"Kamera açılmadı. Brauzerdə kamera icazəsini yoxla.",download:"Şəkli endir",copy:"Kodu kopyala",copied:"Kopyalandı",you:"Siz",examples:["Bir riyaziyyat məsələsi həll et","C++-da sadə oyun yaz","İpək yolu haqqında danış"]},
 tr:{newChat:"＋ Yeni sohbet",history:"Sohbet geçmişi",settings:"⚙ Ayarlar",hero:"Bugün ne yapalım?",intro:"Problem çöz, kod yaz, tarihi araştır veya görsel incele.",remove:"Görseli kaldır",upload:"↑ Görsel yükle",camera:"Kamera",send:"Gönder ↑",accountTitle:"Nexora hesabı",name:"Kullanıcı adı",password:"Şifre (en az 8 karakter)",login:"Giriş yap",register:"Kaydol",close:"Kapat",language:"Dil",theme:"Görünüm",logout:"Çıkış yap",chat:"Sohbet",code:"Kod",image:"Görsel oluştur",dark:"Koyu",light:"Açık",prompt:"Nexora AI’ye yaz…",imagePrompt:"Oluşturulacak görseli tarif et…",wait:"Nexora yanıt hazırlıyor…",badImage:"En fazla 5 MB JPG, PNG veya WebP seç.",cameraError:"Kamera açılamadı. Tarayıcı izinlerini kontrol et.",download:"Görseli indir",copy:"Kodu kopyala",copied:"Kopyalandı",you:"Sen",examples:["Bir matematik problemi çöz","C++ ile basit bir oyun yaz","İpek Yolu’nu anlat"]},
 en:{newChat:"＋ New chat",history:"Chat history",settings:"⚙ Settings",hero:"What shall we do today?",intro:"Solve a problem, write code, explore history, or analyze an image.",remove:"Remove image",upload:"↑ Upload image",camera:"Camera",send:"Send ↑",accountTitle:"Nexora account",name:"Username",password:"Password (at least 8 characters)",login:"Sign in",register:"Register",close:"Close",language:"Language",theme:"Appearance",logout:"Sign out",chat:"Chat",code:"Code",image:"Create image",dark:"Dark",light:"Light",prompt:"Message Nexora AI…",imagePrompt:"Describe the image to create…",wait:"Nexora is working…",badImage:"Choose a JPG, PNG, or WebP up to 5 MB.",cameraError:"Camera unavailable. Check browser permissions.",download:"Download image",copy:"Copy code",copied:"Copied",you:"You",examples:["Solve a math problem","Write a simple game in C++","Tell me about the Silk Road"]}
};
let language = localStorage.getItem("nexoraLanguage") || "az";
if (!Object.hasOwn(words,language)) language="az";
let theme = localStorage.getItem("nexoraTheme") === "light" ? "light" : "dark";
let account = "", current = null, messages = [], attachment = "", stream, busy = false, navigation = 0;
const t = key => words[language][key];
const status = message => by("status").textContent = message;
async function api(path, data){
  const options = data === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)};
  const response = await fetch("/api/"+path,options);
  const result = await response.json();
  if(!response.ok) throw new Error(result.error);
  return result;
}
function select(id, keys, value){
  by(id).replaceChildren(...keys.map(key=>new Option(t(key),key)));
  by(id).value=value;
}
function localize(){
  document.documentElement.lang=language;
  document.documentElement.dataset.theme=theme;
  document.querySelectorAll("[data-label]").forEach(el=>el.textContent=t(el.dataset.label));
  select("theme",["dark","light"],theme);
  select("mode",["chat","code","image"],by("mode").value||"chat");
  by("language").value=language;
  by("examples").replaceChildren(...t("examples").map((example,index)=>{
    const button=document.createElement("button");
    button.textContent=example;
    button.onclick=()=>{by("mode").value=["chat","code","chat"][index];updateMode();by("prompt").value=example;by("prompt").focus()};
    return button;
  }));
  updateMode();
  draw();
}
function updateMode(){
  const imageMode=by("mode").value==="image";
  by("prompt").placeholder=t(imageMode?"imagePrompt":"prompt");
  by("prompt").setAttribute("aria-label",by("prompt").placeholder);
  by("mode").setAttribute("aria-label",t("chat"));
  for(const id of ["upload","cameraButton"]) by(id).disabled=busy||imageMode;
  by("preview").hidden=!attachment||imageMode;
}
function draw(){
  by("account").textContent=account||t("login");
  by("logout").hidden=!account;
  by("landing").hidden=!!messages.length;
  by("messages").replaceChildren(...messages.map(item=>{
    const article=document.createElement("article");
    article.className=item.role;
    const title=document.createElement("strong");
    title.textContent=item.role==="user"?t("you"):"Nexora AI";
    article.append(title);
    if(item.typing){
      const dots=document.createElement("div");
      dots.className="typing";
      dots.setAttribute("role","status");
      dots.setAttribute("aria-label",t("wait"));
      for(let index=0;index<3;index++)dots.append(document.createElement("span"));
      article.append(dots);
    }
    if(item.text){
      const chunks=item.role==="assistant"?item.text.split(/\x60\x60\x60/):[item.text];
      chunks.forEach((chunk,index)=>{
        if(index%2){
          const pre=document.createElement("pre"), code=document.createElement("code"), button=document.createElement("button");
          code.textContent=chunk.replace(/^[\w.+#-]*\r?\n/,"").trimEnd();
          button.textContent=t("copy");
          button.onclick=async()=>{try{await navigator.clipboard.writeText(code.textContent);button.textContent=t("copied")}catch(error){status(error.message)}};
          pre.append(code);article.append(button,pre);
        }else if(chunk.trim()){
          const paragraph=document.createElement("p");
          paragraph.textContent=chunk.trim();article.append(paragraph);
        }
      });
    }
    if(item.image){
      const image=document.createElement("img");image.src=item.image;image.alt=t("image");article.append(image);
      if(item.role==="assistant"){
        const link=document.createElement("a");link.href=item.image;link.download="nexora.png";link.textContent=t("download");article.append(link);
      }
    }
    return article;
  }));
}
async function listChats(){
  const target=account;
  const chats=target?await api("chats"):[];
  if(target!==account)return;
  by("history").replaceChildren(...chats.map(chat=>{
    const button=document.createElement("button");button.textContent=chat.title;
    button.onclick=async()=>{
      if(busy)return;
      const visit=++navigation;
      try{const result=await api("chats/"+chat.id);if(visit!==navigation||busy)return;resetChat();current=chat.id;messages=result;draw()}catch(error){status(error.message)}
    };
    return button;
  }));
}
function resetChat(){
  navigation++;current=null;messages=[];by("prompt").value="";setAttachment("");status("");draw();
}
function setAttachment(image){
  attachment=image;
  if(image)by("attachment").src=image;
  else by("attachment").removeAttribute("src");
  updateMode();
}
by("newChat").onclick=()=>{if(!busy){resetChat();by("prompt").focus()}};
by("account").onclick=()=>by(account?"settings":"auth").showModal();
by("settingsButton").onclick=()=>by("settings").showModal();
document.querySelectorAll("[data-close]").forEach(button=>button.onclick=()=>by(button.dataset.close).close());
by("language").onchange=()=>{language=by("language").value;localStorage.setItem("nexoraLanguage",language);localize()};
by("theme").onchange=()=>{theme=by("theme").value;localStorage.setItem("nexoraTheme",theme);document.documentElement.dataset.theme=theme};
by("mode").onchange=updateMode;
by("authForm").onsubmit=async event=>{
  event.preventDefault();
  const button=event.submitter||by("login");
  by("authError").textContent="";
  by("login").disabled=by("register").disabled=true;
  try{
    const result=await api(button.value,{name:by("name").value,password:by("password").value});
    account=result.name;
    by("authForm").reset();by("auth").close();draw();await listChats();
  }catch(error){if(by("auth").open)by("authError").textContent=error.message;else status(error.message)}
  finally{by("login").disabled=by("register").disabled=false}
};
by("logout").onclick=async()=>{
  if(busy)return;
  try{await api("logout",{});account="";by("settings").close();resetChat();await listChats()}catch(error){status(error.message)}
};
by("upload").onclick=()=>by("file").click();
by("remove").onclick=()=>setAttachment("");
by("file").onchange=()=>{
  const file=by("file").files[0];by("file").value="";
  if(!file)return;
  if(!["image/jpeg","image/png","image/webp"].includes(file.type)||file.size>5*1024*1024){status(t("badImage"));return}
  const reader=new FileReader();
  reader.onload=()=>{if(!busy)setAttachment(reader.result)};
  reader.onerror=()=>status(t("badImage"));
  reader.readAsDataURL(file);
};
by("cameraButton").onclick=async()=>{
  by("camera").showModal();by("snap").disabled=true;
  try{
    const camera=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280}},audio:false});
    if(!by("camera").open){camera.getTracks().forEach(track=>track.stop());return}
    stream=camera;by("video").srcObject=stream;
  }catch(error){if(by("camera").open)by("camera").close();status(t("cameraError"))}
};
by("video").onloadeddata=()=>by("snap").disabled=!by("camera").open;
by("camera").onclose=()=>{if(stream)stream.getTracks().forEach(track=>track.stop());stream=null;by("video").srcObject=null};
by("snap").onclick=()=>{
  const video=by("video");if(!video.videoWidth)return;
  const canvas=document.createElement("canvas");
  const scale=Math.min(1,1600/Math.max(video.videoWidth,video.videoHeight));
  canvas.width=Math.round(video.videoWidth*scale);canvas.height=Math.round(video.videoHeight*scale);
  canvas.getContext("2d").drawImage(video,0,0,canvas.width,canvas.height);
  setAttachment(canvas.toDataURL("image/jpeg",0.88));by("camera").close();
};
by("prompt").onkeydown=event=>{
  if(event.key==="Enter"&&!event.shiftKey&&!event.isComposing){event.preventDefault();by("composer").requestSubmit()}
};
by("composer").onsubmit=async event=>{
  event.preventDefault();
  if(busy)return;
  const draft=by("prompt").value, text=draft.trim(), mode=by("mode").value;
  const image=mode==="image"?"":attachment;
  if(!text&&!image)return;
  if(!account){by("auth").showModal();return}
  busy=true;navigation++;
  const previous=messages, draftImage=attachment;
  messages=[...previous,{role:"user",text,image},{role:"assistant",typing:true}];
  by("prompt").value="";
  for(const id of ["send","prompt","mode","newChat"])by(id).disabled=true;
  setAttachment("");
  status("");
  draw();
  window.scrollTo(0,document.body.scrollHeight);
  try{
    const result=await api("send",{id:current,text,image,mode,language});
    current=result.id;messages=result.messages;
    if(mode==="image")setAttachment(draftImage);
    draw();
    window.scrollTo(0,document.body.scrollHeight);
    listChats().catch(error=>status(error.message));
  }catch(error){
    messages=previous;by("prompt").value=draft;setAttachment(draftImage);
    draw();status(error.message);
  }finally{
    busy=false;
    for(const id of ["send","prompt","mode","newChat"])by(id).disabled=false;
    updateMode();by("prompt").focus();
  }
};
localize();
api("me").then(result=>{account=result.name;draw();return listChats()}).catch(error=>status(error.message));
</script></body></html>"""


def main():
    initialize()
    with ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("NEXORA_PORT", "8765"))), Handler) as server:
        address = "http://127.0.0.1:" + str(server.server_port)
        print("Nexora AI:", address, flush=True)
        print("Keep this window open. Press Ctrl+C to stop.", flush=True)
        threading.Thread(target=webbrowser.open, args=(address,), daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
