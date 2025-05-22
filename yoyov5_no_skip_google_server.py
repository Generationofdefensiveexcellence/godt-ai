import os
import cv2
import base64
import random
import threading
import numpy as np
import uuid
import subprocess
import secrets
from datetime import datetime, timedelta
import pandas as pd
import pytesseract  # For jersey number detection
import openai       # For conversational coaching
from sklearn.linear_model import LinearRegression

from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, jsonify, session
from flask_mail import Mail, Message
from ultralytics import YOLO

# ========================
# CONFIGURATION & ASSETS
# ========================

# --- OpenAI API key (replace with your key)
openai.api_key = "your-openai-api-key-here"

# --- Directories and Log Files
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
PRO_VIDEOS_DIR = os.path.join(os.getcwd(), "pro_videos")
os.makedirs(PRO_VIDEOS_DIR, exist_ok=True)
LOG_DIR = os.path.join(os.getcwd(), "SportsAI")
os.makedirs(LOG_DIR, exist_ok=True)

ADMIN_LOG_FILE = os.path.join(LOG_DIR, "admin_logs.txt")
USER_LOG_FILE = os.path.join(LOG_DIR, "user_logs.txt")
SESSION_LOG_FILE = os.path.join(LOG_DIR, "session_details.txt")

# --- Global In‑Memory Data Stores
PENDING_REQUESTS = {}       # {username: {...}}
APPROVED_REQUESTS = {}      # {username: {...}}
ADMIN_CHECKINS = {}         # {admin_username: {...}}
USER_TRACKING_DATA = {}     # {username: {...}}
GLOBAL_PROGRESS = {}        # {uid: progress int}
GLOBAL_LIVE_FRAME = {}      # {uid: current frame (cv2 image)}
GLOBAL_EVENT_FRAMES = {}    # {uid: list of (event_type, encoded image)}
GLOBAL_RESULTS = {}         # {uid: {videos:[], ...}}
CHAT_MESSAGES = {}          # {uid: [(from, message), ...]}
LIVE_CHAT = {}              # {username: {active:bool, messages:list, ended:bool, ended_at:str}}

ADMIN_LOGS = []
USER_LOGS = []

# --- Static Data & Configuration
ATTRIBUTES = ["Speed", "Strength", "Awareness", "Tackling", "Agility",
              "Stamina", "Positioning", "Interceptions", "Leadership", "Vision"]
STAT_RANGES = {
    "Speed": (65, 90), "Strength": (50, 90), "Awareness": (40, 95),
    "Tackling": (60, 95), "Agility": (50, 90), "Stamina": (50, 95),
    "Positioning": (40, 90), "Interceptions": (50, 85), "Leadership": (40, 100),
    "Vision": (50, 95)
}
COLOR_OPTIONS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "gray"]
NATIONALITIES = ["England", "France", "Brazil", "Nigeria", "USA", "Spain", "Germany", "Argentina", "Others"]

TRIVIA_LINES = [
    "Catenaccio revolutionized defensive tactics in the 1960s!",
    "A single perfect tackle can shift a match's momentum!",
    "Elite defenders blend speed, strength, and tactical brilliance!",
    "Zonal marking is a cornerstone of modern defenses!",
    "Reading the game is as crucial as raw athleticism!",
    "High-press relies on defenders who anticipate passes instantly!",
    "Great defenders coordinate the back line like chess masters!",
    "A timely interception can spark a lethal counterattack!"
]
LONG_SPEECHES = [
    "Remember: The pitch is your battlefield. As a defender, you're the shield between the opposition and victory!",
    "In the heat of competition, a defender's resolve can shatter an opponent's spirit!",
    "Stay disciplined and aware—communication and positioning are key to an impenetrable defense!"
]

BRAND_IMAGE = "stopping_mbappe_logo.jpg"  # Ensure this image exists
KEEPER_IMAGE = "keeper.png"               # Image for the mini-game

# --- YOLO MODEL: Use YOLOv8 (official, lightweight model)
YOLO_MODEL_PATH = "yolov8s.pt"
YOLO_CONF_THRESHOLD = 0.5
COLLISION_THRESHOLDS = {1: 2000, 2: 8000}

SECRET_TOKEN = "globaltest2025"

# ========================
# FLASK APP CONFIGURATION
# ========================

app = Flask("GODT")
app.secret_key = secrets.token_hex(16)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=False  # Change to True in production
)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "AdminRocks")
app.config.update(
    MAIL_SERVER="smtp.gmail.com",
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USE_SSL=False,
    MAIL_USERNAME=os.getenv("MAIL_USERNAME", "projectstoppingmbappes@gmail.com"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD", "your-mail-app-password"),
    MAIL_DEFAULT_SENDER=os.getenv("MAIL_DEFAULT_SENDER", "projectstoppingmbappes@gmail.com")
)
mail = Mail(app)

ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov'}

def allowed_file(filename):
    return '.' in filename and os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

# ------------------------
# Load YOLOv8 Model (Auto-downloads if needed)
# ------------------------
global_yolo_model = YOLO(YOLO_MODEL_PATH)

# ------------------------
# Middleware: Access Control
# ------------------------
@app.before_request
def restrict_access():
    allowed_paths = ["/serve_audio", "/serve_bg_image", "/serve_keeper_image", "/static/", "/user_login", "/coach_chat"]
    if any(request.path.startswith(p) for p in allowed_paths):
        return
    token = request.args.get("token") or session.get("access_token")
    if token != SECRET_TOKEN:
        return "Access Denied. Please use the provided link.", 403
    session["access_token"] = token

# ------------------------
# Logging Helper
# ------------------------
def add_log(log_list, message, log_type="general"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    log_list.append(log_entry)
    if log_type == "admin":
        with open(ADMIN_LOG_FILE, "a") as f:
            f.write(log_entry + "\n")
    elif log_type == "user":
        with open(USER_LOG_FILE, "a") as f:
            f.write(log_entry + "\n")
    else:
        with open(SESSION_LOG_FILE, "a") as f:
            f.write(log_entry + "\n")

# ========================
# MEMORY SYSTEM: Save & Compare Player Data
# ========================
def store_player_data(player_id, session_data):
    file_path = f"player_{player_id}_performance.csv"
    session_df = pd.DataFrame(session_data)
    if os.path.exists(file_path):
        session_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        session_df.to_csv(file_path, index=False)

def compare_performance(player_id, current_session):
    file_path = f"player_{player_id}_performance.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        previous_session = df.tail(1)
        change = current_session - previous_session.iloc[0]
        return change
    return None

# ========================
# PREDICTIVE SIMULATION (Placeholder)
# ========================
def predict_player_growth(X_train, y_train, current_stats, training_plan_vector):
    model = LinearRegression()
    model.fit(np.array(X_train), np.array(y_train))
    predicted_stats = model.predict([current_stats + training_plan_vector])
    return predicted_stats[0]

# ========================
# CONVERSATIONAL COACHING ENDPOINT
# ========================
@app.route("/coach_chat", methods=["POST"])
def coach_chat():
    question = request.form.get("question", "")
    context = "Your recent performance and training history are available."
    prompt = f"Act as a professional defensive coach. Context: {context}\nPlayer Question: {question}\nAdvice:"
    try:
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            temperature=0.5,
            max_tokens=150
        )
        answer = response.choices[0].text.strip()
    except Exception as e:
        answer = "Sorry, I couldn't process your question right now."
    return jsonify({"response": answer})

# ========================
# LIVE CHAT ENDPOINTS (User and Admin)
# ========================
@app.route("/user_live_chat")
def user_live_chat():
    username = session.get("username")
    if not username:
        return jsonify({"active": False})
    chat = LIVE_CHAT.get(username, None)
    return jsonify(chat if chat else {"active": False})

@app.route("/admin_live_chat/<target_username>")
def admin_live_chat(target_username):
    chat = LIVE_CHAT.get(target_username, None)
    return jsonify(chat if chat else {"active": False})

@app.route("/admin_initiate_chat/<target_username>")
def admin_initiate_chat(target_username):
    if not admin_session_valid():
        return redirect(url_for("admin_option", token=SECRET_TOKEN))
    if target_username in APPROVED_REQUESTS:
        LIVE_CHAT[target_username] = {"active": True, "messages": [("admin", "Hello, how can I help you?")], "ended": False}
        add_log(ADMIN_LOGS, f"Admin initiated live chat with user '{target_username}'.", "admin")
        return redirect(url_for("admin_chat", target_username=target_username, token=SECRET_TOKEN))
    return f"User {target_username} not found."

@app.route("/admin_chat/<target_username>")
def admin_chat(target_username):
    if not admin_session_valid():
        return redirect(url_for("admin_option", token=SECRET_TOKEN))
    html = """
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Chat with {{target_username}}</title>
      <style>
        body { background: #222; color: #fff; font-family: Arial; text-align: center; padding: 20px; }
        #chatMessages { max-height: 300px; overflow-y: auto; border: 1px solid #555; padding: 10px; margin-bottom: 10px; }
        input[type=text] { width: 70%; padding: 8px; margin-bottom: 10px; }
        button { padding: 8px 16px; background: #0af; border: none; border-radius: 4px; color: #fff; cursor: pointer; }
      </style>
      <script>
        function pollChat(){
          fetch('/admin_live_chat/{{target_username}}?token={{token}}')
            .then(r => r.json())
            .then(data => {
              var chatDiv = document.getElementById("chatMessages");
              chatDiv.innerHTML = "";
              if(data.messages){
                data.messages.forEach(function(msg){
                  chatDiv.innerHTML += "<p><strong>" + msg[0] + ":</strong> " + msg[1] + "</p>";
                });
              }
              if(data.ended){
                window.location.href = "/admin_dashboard?token={{token}}";
              } else {
                setTimeout(pollChat, 3000);
              }
            });
        }
        function sendAdminChat(){
          var msg = document.getElementById("adminChatInput").value;
          fetch('/admin_send_chat/{{target_username}}?token={{token}}', {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: 'message=' + encodeURIComponent(msg)
          }).then(r => r.text()).then(function(){
            document.getElementById("adminChatInput").value = "";
          });
        }
        function endChat(){
          fetch('/admin_end_chat/{{target_username}}?token={{token}}')
            .then(r => r.text())
            .then(function(){
              window.location.href = "/admin_dashboard?token={{token}}";
            });
        }
        window.onload = function(){ pollChat(); }
      </script>
    </head>
    <body>
      <h1>Chat with {{target_username}}</h1>
      <div id="chatMessages"></div>
      <input type="text" id="adminChatInput" placeholder="Type your message..."/>
      <button onclick="sendAdminChat()">Send</button>
      <br/><br/>
      <button onclick="endChat()">End Chat</button>
      <p><a href="/admin_dashboard?token={{token}}">Back to Dashboard</a></p>
    </body>
    </html>
    """
    return render_template_string(html, target_username=target_username, token=SECRET_TOKEN)

@app.route("/admin_send_chat/<target_username>", methods=["POST"])
def admin_send_chat(target_username):
    if not admin_session_valid():
        return redirect(url_for("admin_option", token=SECRET_TOKEN))
    message = request.form.get("message", "")
    if target_username in LIVE_CHAT and LIVE_CHAT[target_username]["active"]:
        LIVE_CHAT[target_username]["messages"].append(("admin", message))
        add_log(ADMIN_LOGS, f"Admin sent message to '{target_username}': {message}", "admin")
        return "Message sent."
    return "No active chat with this user."

@app.route("/admin_end_chat/<target_username>")
def admin_end_chat(target_username):
    if not admin_session_valid():
        return redirect(url_for("admin_option", token=SECRET_TOKEN))
    if target_username in LIVE_CHAT and LIVE_CHAT[target_username]["active"]:
        LIVE_CHAT[target_username]["active"] = False
        LIVE_CHAT[target_username]["ended"] = True
        LIVE_CHAT[target_username]["ended_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        add_log(ADMIN_LOGS, f"Admin ended live chat with user '{target_username}'.", "admin")
    return redirect(url_for("admin_dashboard", token=SECRET_TOKEN))

# ========================
# ASSET SERVING ENDPOINTS
# ========================
@app.route("/serve_audio/<filename>")
def serve_audio(filename):
    return send_from_directory(os.getcwd(), filename)

@app.route("/serve_bg_image")
def serve_bg_image():
    return send_from_directory(os.getcwd(), BRAND_IMAGE)

@app.route("/serve_keeper_image")
def serve_keeper_image():
    return send_from_directory(os.getcwd(), KEEPER_IMAGE)

# ========================
# MAIN PAGE (No Go Pro Button)
# ========================
@app.route("/")
def main_page():
    html = """
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Main Page</title>
      <link rel="manifest" href="/static/manifest.json">
      <meta name="theme-color" content="#28a745">
      <style>
        body { background: linear-gradient(135deg, #0d0d0d, #333); color: #fff; font-family: "Trebuchet MS", sans-serif; text-align: center; padding:50px; }
        .btn { display:inline-block; padding:15px 30px; margin:20px; background:#28a745; color:#fff; text-decoration:none; border-radius:8px; transition: box-shadow 0.3s; }
        .btn:hover { box-shadow: 0 0 12px rgba(40,167,69,0.7); }
        h1 { text-shadow: 0 0 10px #0f0; }
      </style>
      <script>
        if ('serviceWorker' in navigator) {
          window.addEventListener('load', function() {
            navigator.serviceWorker.register('/static/service-worker.js').then(function(registration) {
              console.log('ServiceWorker registration successful with scope: ', registration.scope);
            }, function(err) {
              console.log('ServiceWorker registration failed: ', err);
            });
          });
        }
      </script>
    </head>
    <body>
      <h1>Welcome to GODT Generation of Defensive Talent</h1>
      <a class="btn" href="/admin_option?token={{token}}">Admin Dashboard</a>
      <a class="btn" href="/analysis_form?token={{token}}">Analysis Page</a>
      <a class="btn" href="/user_login?token={{token}}">Existing User Login</a>
    </body>
    </html>
    """
    return render_template_string(html, token=SECRET_TOKEN)

# ========================
# ADMIN LOGIN & DASHBOARD
# ========================
@app.route("/admin_option")
def admin_option():
    if session.get("admin_logged_in"):
        return redirect(url_for("admin_dashboard", token=SECRET_TOKEN))
    html = """
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Admin Login</title>
      <link rel="manifest" href="/static/manifest.json">
      <meta name="theme-color" content="#28a745">
      <style>
        body { background: radial-gradient(circle, #000, #222); color:#fff; text-align:center; font-family:Arial; padding:50px; }
        .box { background:rgba(0,0,0,0.7); width:300px; margin:0 auto; padding:20px; border-radius:8px; }
        input { width:80%; margin:5px 0; padding:8px; border:none; border-radius:4px; }
        button { padding:10px 20px; border:none; border-radius:4px; background:#f00; color:#fff; cursor:pointer; transition: box-shadow 0.3s; }
        button:hover { box-shadow: 0 0 10px #f00; }
        h1 { text-shadow: 0 0 8px #f00; }
        a { color:#0af; text-decoration:none; }
      </style>
    </head>
    <body>
      <h1>Admin Login</h1>
      <div class="box">
        <form method="POST" action="/admin_login?token={{token}}">
          <input type="text" name="username" placeholder="Admin Username" required/><br/>
          <input type="password" name="password" placeholder="Password" required/><br/>
          <button type="submit">Login as Admin</button>
        </form>
      </div>
      <p><a href="/?token={{token}}" style="color:#0af;">Back to Main Page</a></p>
    </body>
    </html>
    """
    return render_template_string(html, token=SECRET_TOKEN)

@app.route("/admin_login", methods=["POST"])
def admin_login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    if username == "admin" and password == ADMIN_PASSWORD:
        session["admin_logged_in"] = True
        session["admin_username"] = username
        session['expires_at'] = (datetime.utcnow() + timedelta(hours=2)).isoformat()
        add_log(ADMIN_LOGS, f"Admin '{username}' logged in.", "admin")
        return redirect(url_for("admin_dashboard", token=SECRET_TOKEN))
    error_html = """
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Windows Security</title>
      <style>
        body { font-family:Arial; background:#f0f0f0; color:#333; text-align:center; padding-top:100px; }
        .error-box { background:#fff; border:1px solid #ccc; display:inline-block; padding:20px; box-shadow: 0 0 10px rgba(0,0,0,0.2); }
        h2 { color:#f00; }
        a { text-decoration:none; color:#00f; }
      </style>
    </head>
    <body>
      <div class="error-box">
        <h2>Windows Security</h2>
        <p>Invalid credentials provided. Please try again.</p>
        <p><a href="/admin_option?token={{token}}">Back to Login</a></p>
      </div>
    </body>
    </html>
    """
    return render_template_string(error_html, token=SECRET_TOKEN)

def admin_session_valid():
    if not session.get("admin_logged_in"):
        return False
    if 'expires_at' in session:
        if datetime.utcnow() > datetime.fromisoformat(session['expires_at']):
            session.clear()
            return False
    return True

@app.route("/admin_dashboard")
def admin_dashboard():
    if not admin_session_valid():
        return redirect(url_for("admin_option", token=SECRET_TOKEN))
    admin_username = session.get("admin_username", "unknown")
    checkin_status = session.get("admin_checked_in", False)
    checkin_button = (
        f'<a class="btn" href="/admin_checkin?token={SECRET_TOKEN}" style="background:#28a745;">Check In</a>'
        if not checkin_status else 
        f'<a class="btn" href="/admin_checkout?token={SECRET_TOKEN}" style="background:#f00;">Check Out</a>'
    )
    num_admins = len(ADMIN_CHECKINS)
    num_users = len(APPROVED_REQUESTS)
    active_chats = LIVE_CHAT
    html = f"""
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Admin Dashboard</title>
      <link rel="manifest" href="/static/manifest.json">
      <meta name="theme-color" content="#28a745">
      <style>
        body {{
          background: linear-gradient(135deg, #111, #333);
          color:#fff; text-align:center; font-family:Arial; padding:20px;
        }}
        h1, h2 {{ text-shadow: 0 0 5px #0f0; }}
        a {{ color:#0af; text-decoration:none; }}
        .btn {{ padding:10px 20px; background:#28a745; color:#fff; border:none; border-radius:4px; margin:10px; cursor:pointer; transition: box-shadow 0.3s; }}
        .btn:hover {{ box-shadow: 0 0 12px rgba(40,167,69,0.7); }}
        table {{ margin:10px auto; border-collapse: collapse; width:95%; }}
        th, td {{ padding:10px; border:1px solid #555; }}
        th {{ background: #222; }}
        .action-btn {{ font-weight: bold; font-size: 1.1em; }}
        #admin_logs, #user_logs {{
          background:rgba(0,0,0,0.8); max-height:200px; overflow-y:scroll; margin:10px auto; padding:10px; border-radius:6px;
          text-align:left; width:80%; box-shadow: inset 0 0 10px #444;
        }}
        .admin-profile {{
          width: 80px; height: 80px; border-radius: 50%; background: #555;
          display: inline-block; vertical-align: middle; margin-right: 10px;
          background-image: url('/serve_bg_image'); background-size: cover;
        }}
        #initiateChat {{
          margin-top: 20px; padding: 10px; background: #444; border-radius: 8px; display: inline-block;
        }}
        #initiateChat input {{ padding: 8px; border: none; border-radius: 4px; margin-right: 10px; }}
        #initiateChat button {{ padding: 8px 16px; border: none; border-radius: 4px; background: #0af; color: #fff; font-weight: bold; cursor: pointer; }}
      </style>
      <script>
        function fetchLogs() {{
          fetch('/admin_logs?token={SECRET_TOKEN}').then(r => r.json()).then(data => {{
              document.getElementById('admin_logs').innerHTML = data.logs.join("<br/>");
          }});
          fetch('/user_logs?token={SECRET_TOKEN}').then(r => r.json()).then(data => {{
              document.getElementById('user_logs').innerHTML = data.logs.join("<br/>");
          }});
          setTimeout(function(){{ location.reload(); }}, 50000);
        }}
        setInterval(fetchLogs, 50000);
        function initiateLiveChat() {{
          var targetUser = document.getElementById("chatUsername").value;
          if(targetUser){{
            window.location.href = "/admin_initiate_chat/" + targetUser + "?token={SECRET_TOKEN}";
          }} else {{
            alert("Please enter a username to initiate chat.");
          }}
        }}
      </script>
    </head>
    <body onload="fetchLogs()">
      <div style="display:flex; align-items:center; justify-content:center;">
        <div class="admin-profile"></div>
        <h1>Admin Dashboard</h1>
      </div>
      <p>Logged in as admin (<strong>{admin_username}</strong>)</p>
      <p>{checkin_button}</p>
      <p><strong>Online:</strong> Admins: {num_admins} | Users: {num_users}</p>
      <h2>Pending/Approved User Requests</h2>
      <table>
        <tr>
          <th>Username</th>
          <th>Player Name</th>
          <th>Check-In Time</th>
          <th># Videos</th>
          <th>Status</th>
          <th>IP Address</th>
          <th>Action</th>
        </tr>
        {"".join([
            f"<tr><td>{uname}</td><td>{data.get('player_name','')}</td><td>{data.get('checkin_time','N/A')}</td>"
            f"<td>{len(data.get('videos', []))}</td><td>{'Approved' if data.get('approved', False) else 'Pending'}</td>"
            f"<td>{data.get('ip_address','Unknown')}</td><td>" +
            (f"<a href='/approve/{uname}?token={SECRET_TOKEN}' class='action-btn' style='color:#0f0;'>Approve</a> | "
             f"<a href='/reject/{uname}?token={SECRET_TOKEN}' class='action-btn' style='color:#f00;'>Reject</a>" 
             if not data.get('approved', False) else "-")
            + "</td></tr>"
            for uname, data in list(PENDING_REQUESTS.items()) + list(APPROVED_REQUESTS.items())
        ])}
      </table>
      <h2>Active Live Chats</h2>
      <div id="active_chats">
        {"".join([
          f"<div style='border:1px solid #fff; padding:10px; margin:5px;'><strong>User: {uname}</strong> - Active: {chat['active']}<br/>"
          f"<a class='btn action-btn' href='/admin_initiate_chat/{uname}?token={SECRET_TOKEN}'>Open Chat</a> | "
          f"<a class='btn action-btn' href='/admin_end_chat/{uname}?token={SECRET_TOKEN}'>End Chat &amp; Return</a></div>"
          for uname, chat in active_chats.items()
        ])}
      </div>
      <div id="initiateChat">
        <input type="text" id="chatUsername" placeholder="Enter username to chat"/>
        <button onclick="initiateLiveChat()">Initiate Live Chat</button>
      </div>
      <h2>Live Admin Logs</h2>
      <div id="admin_logs"></div>
      <h2>Live User Logs</h2>
      <div id="user_logs"></div>
      <p>
        <a href="/admin_logout?token={SECRET_TOKEN}" class="btn" style="background:#555;">Logout Admin</a> | 
        <a href="/?token={SECRET_TOKEN}" style="color:#0af;">Back to Main Page</a>
      </p>
    </body>
    </html>
    """
    return html

@app.route("/admin_checkin")
def admin_checkin():
    if not admin_session_valid():
        return redirect(url_for("admin_option", token=SECRET_TOKEN))
    admin_username = session.get("admin_username")
    session["admin_checked_in"] = True
    checkin_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ADMIN_CHECKINS[admin_username] = {"checkin_time": checkin_time}
    add_log(ADMIN_LOGS, f"Admin '{admin_username}' checked in at {checkin_time}.", "admin")
    return redirect(url_for("admin_dashboard", token=SECRET_TOKEN))

@app.route("/admin_checkout", methods=["GET", "POST"])
def admin_checkout():
    if not admin_session_valid() or not session.get("admin_checked_in"):
        return redirect(url_for("admin_option", token=SECRET_TOKEN))
    if request.method == "POST":
        feedback = request.form.get("feedback", "").replace("\n", " ").strip()
        admin_username = session.get("admin_username")
        add_log(ADMIN_LOGS, f"Admin '{admin_username}' checked out with feedback: {feedback}", "admin")
        session.pop("admin_checked_in", None)
        session.pop("admin_logged_in", None)
        session.pop("admin_username", None)
        return redirect(url_for("main_page", token=SECRET_TOKEN))
    html = """
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Admin Check-Out</title>
      <style>
        body { background: linear-gradient(135deg, #111, #333); color:#fff; text-align:center; font-family:Arial; padding:50px; }
        textarea { width:80%; padding:8px; margin:10px; border:none; border-radius:4px; }
        button { padding:10px 20px; background:#f00; color:#fff; border:none; border-radius:4px; cursor:pointer; transition: box-shadow 0.3s; }
        button:hover { box-shadow: 0 0 10px #f00; }
        h1 { text-shadow: 0 0 5px #f00; }
      </style>
    </head>
    <body>
      <h1>Admin Check-Out</h1>
      <form method="POST" action="/admin_checkout?token={{token}}">
         <textarea name="feedback" placeholder="How did your day go? (Optional)" rows="4"></textarea><br/>
         <button type="submit">Check Out</button>
      </form>
      <p><a href="/admin_dashboard?token={{token}}" style="color:#0af;">Back to Dashboard</a></p>
    </body>
    </html>
    """
    return render_template_string(html, token=SECRET_TOKEN)

@app.route("/approve/<uname>")
def approve(uname):
    if not admin_session_valid():
        return redirect(url_for("admin_option", token=SECRET_TOKEN))
    if uname in PENDING_REQUESTS:
        PENDING_REQUESTS[uname]["approved"] = True
        APPROVED_REQUESTS[uname] = PENDING_REQUESTS.pop(uname)
        USER_TRACKING_DATA[uname] = {
            "player_name": APPROVED_REQUESTS[uname].get("player_name", "Defender"),
            "jersey_number": APPROVED_REQUESTS[uname].get("jersey_number", "10"),
            "nationality": APPROVED_REQUESTS[uname].get("nationality", "Others")
        }
        referrer = APPROVED_REQUESTS[uname].get("referrer", "")
        if referrer and referrer in APPROVED_REQUESTS:
            APPROVED_REQUESTS[referrer]["upload_credit"] += 5
        add_log(USER_LOGS, f"User '{uname}' approved by admin.", "user")
    return redirect(url_for("admin_dashboard", token=SECRET_TOKEN))

@app.route("/reject/<uname>")
def reject(uname):
    if not admin_session_valid():
        return redirect(url_for("admin_option", token=SECRET_TOKEN))
    if uname in PENDING_REQUESTS:
        PENDING_REQUESTS.pop(uname)
        add_log(USER_LOGS, f"User '{uname}' was rejected by admin.", "user")
    return redirect(url_for("admin_dashboard", token=SECRET_TOKEN))

@app.route("/admin_logout")
def admin_logout():
    admin = session.get("admin_username", "unknown")
    add_log(ADMIN_LOGS, f"Admin '{admin}' logged out.", "admin")
    session.clear()
    return redirect(url_for("main_page", token=SECRET_TOKEN))

@app.route("/admin_logs")
def admin_logs():
    return jsonify({"logs": ADMIN_LOGS[-50:]})

@app.route("/user_logs")
def user_logs():
    return jsonify({"logs": USER_LOGS[-50:]})

# ========================
# ANALYSIS FORM (User Registration & Video Upload)
# ========================
@app.route("/analysis_form", methods=["GET", "POST"])
def analysis_form():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        player_name = request.form.get("player_name", "Defender").strip()
        age = request.form.get("player_age", "26")
        jersey_number = request.form.get("jersey_number", "10")
        nationality = request.form.get("nationality", "Others")
        jersey_colors = request.form.getlist("jersey_colors")
        boots_colors = request.form.getlist("boots_colors")
        files = request.files.getlist("upload_files")
        if not username or not email:
            return "<h3>Username and Email are required!</h3><a href='/analysis_form?token={token}'>Go Back</a>".format(token=SECRET_TOKEN)
        if username in PENDING_REQUESTS or username in APPROVED_REQUESTS:
            return "<h3>Username already exists! Choose a different username.</h3><a href='/analysis_form?token={token}'>Go Back</a>".format(token=SECRET_TOKEN)
        if not files or len(files) == 0:
            return "<h3>You must upload at least one video!</h3><a href='/analysis_form?token={token}'>Go Back</a>".format(token=SECRET_TOKEN)
        if len(files) > 7:
            return "<h3>Maximum of 7 videos allowed at once!</h3><a href='/analysis_form?token={token}'>Go Back</a>".format(token=SECRET_TOKEN)
        user_uid = str(uuid.uuid4())[:8]
        video_files = []
        for f in files:
            if f.filename and allowed_file(f.filename):
                filename = f"{username}_{f.filename}"
                sp = os.path.join(UPLOAD_DIR, filename)
                f.save(sp)
                video_files.append(filename)
            else:
                return "<h3>Invalid file type uploaded.</h3><a href='/analysis_form?token={token}'>Go Back</a>".format(token=SECRET_TOKEN)
        upload_credit = 7 - len(video_files)
        referrer = session.get("referrer", "")
        ip_address = request.remote_addr or "Unknown"
        add_log(USER_LOGS, f"User '{username}' submitted analysis form with {len(video_files)} videos from IP {ip_address}.", "user")
        PENDING_REQUESTS[username] = {
            "uid": user_uid,
            "email": email,
            "player_name": player_name,
            "age": age,
            "jersey_number": jersey_number,
            "nationality": nationality,
            "jersey_colors": jersey_colors,
            "boots_colors": boots_colors,
            "videos": video_files,
            "checkin_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "approved": False,
            "upload_credit": upload_credit,
            "referrer": referrer,
            "ip_address": ip_address
        }
        session["username"] = username
        html = f"""
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Request Submitted</title>
          <style>
            body {{
              background: radial-gradient(circle, #111, #333);
              color:#fff; text-align:center; font-family:Arial; padding:50px;
            }}
            a {{ color:#0af; text-decoration:none; }}
            h2 {{ text-shadow: 0 0 5px #0f0; }}
          </style>
        </head>
        <body>
          <h2>Your request has been submitted and is pending admin approval.</h2>
          <p>Your upload credit is: {upload_credit} (7 initial minus videos uploaded)</p>
          <p>Please wait until an admin approves your request.</p>
          <script>
            setInterval(function(){{
              fetch('/user_status?token={SECRET_TOKEN}')
                .then(r => r.text())
                .then(html => {{
                  if(html.indexOf("Approved") !== -1) {{
                    window.location.href = "/confirm_upload?token={SECRET_TOKEN}";
                  }}
                }});
            }}, 5000);
          </script>
          <p><a href="/user_status?token={SECRET_TOKEN}">Check Your Status</a></p>
          <p><a href="/?token={SECRET_TOKEN}">Back to Main Page</a></p>
        </body>
        </html>
        """
        return html
    form_html = """
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Analysis Form</title>
      <link rel="manifest" href="/static/manifest.json">
      <meta name="theme-color" content="#28a745">
      <style>
        body { background: url('/serve_bg_image') no-repeat center center; background-size: cover; color: #fff; text-align: center; font-family: Arial; padding: 20px; }
        .overlay { width: 600px; margin: 50px auto; background: rgba(0,0,0,0.7); padding: 20px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.5); }
        label { display: block; margin-top: 10px; text-align: left; }
        input, select { width: 100%; padding: 8px; margin-top: 5px; border: none; border-radius: 4px; }
        button { margin-top: 20px; padding: 10px; width: 100%; border: none; border-radius: 4px; cursor: pointer; background: #28a745; color: #fff; font-weight: bold; transition: box-shadow 0.3s; }
        button:hover { box-shadow: 0 0 10px #28a745; }
        h1 { text-shadow: 0 0 6px #0f0; }
        .welcome { font-size: 1.5em; color: #ff0; margin-bottom: 20px; }
        a { color: #0af; text-decoration: none; }
      </style>
    </head>
    <body>
      <div class="overlay">
        <div class="welcome">Behold… your awakening has begun!</div>
        <h1>Enter Your Details for Analysis</h1>
        <form method="POST" enctype="multipart/form-data">
          <label>Username (unique):</label>
          <input type="text" name="username" required/>
          <label>Email Address:</label>
          <input type="email" name="email" required/>
          <label>Player Name:</label>
          <input type="text" name="player_name" required/>
          <label>Age (6-35):</label>
          <input type="number" name="player_age" min="6" max="35" required/>
          <label>Jersey Number (0-100):</label>
          <input type="number" name="jersey_number" min="0" max="100" required/>
          <label>Nationality:</label>
          <select name="nationality" required>
            {% for nat in nationalities %}
              <option value="{{nat}}">{{nat}}</option>
            {% endfor %}
          </select>
          <label>Jersey Colors (Ctrl+click to select multiple):</label>
          <select name="jersey_colors" multiple required>
            {% for col in color_options %}
              <option value="{{col}}">{{col}}</option>
            {% endfor %}
          </select>
          <label>Boots Colors (Ctrl+click):</label>
          <select name="boots_colors" multiple required>
            {% for col in color_options %}
              <option value="{{col}}">{{col}}</option>
            {% endfor %}
          </select>
          <label>Upload up to 7 Videos:</label>
          <input type="file" name="upload_files" multiple accept=".mp4,.avi,.mov" required/>
          <button type="submit">Submit Request</button>
        </form>
        <p><a href="/?token={{token}}">Back to Main Page</a></p>
      </div>
    </body>
    </html>
    """
    return render_template_string(form_html, nationalities=NATIONALITIES, color_options=COLOR_OPTIONS, token=SECRET_TOKEN)

# ========================
# USER STATUS PAGE
# ========================
@app.route("/user_status")
def user_status():
    username = session.get("username")
    if not username:
        return redirect(url_for("analysis_form", token=SECRET_TOKEN))
    if username in APPROVED_REQUESTS:
        credit = APPROVED_REQUESTS[username].get("upload_credit", 0)
        referral_link = f"{request.host_url}?ref={username}&token={SECRET_TOKEN}"
        account_type = APPROVED_REQUESTS[username].get("account_type", "basic")
        pro_notice = " (Unlimited Uploads)" if account_type == "pro" else ""
        html = f"""
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Status: Approved</title>
          <link rel="manifest" href="/static/manifest.json">
          <meta name="theme-color" content="#28a745">
          <style>
            body {{
              background: linear-gradient(135deg, #111, #333);
              color: #fff; text-align: center; font-family: Arial; padding: 50px;
            }}
            h2 {{ text-shadow: 0 0 5px #0f0; }}
            a {{ color: #0af; text-decoration: none; }}
            input[type=text] {{ width: 80%; padding: 8px; }}
            button {{
              padding: 10px 20px; background: #28a745; color: #fff; border: none; border-radius: 4px; cursor: pointer;
              transition: box-shadow 0.3s;
            }}
            button:hover {{ box-shadow: 0 0 10px #28a745; }}
            .btn {{
              display: inline-block; padding: 10px 20px; margin: 10px;
              background: #0af; color: #fff; border: none; border-radius: 4px; text-decoration: none;
              transition: box-shadow 0.3s;
            }}
            .btn:hover {{ box-shadow: 0 0 10px #0af; }}
          </style>
          <script>
            setTimeout(function(){{
              window.location.href = "/confirm_upload?token={SECRET_TOKEN}";
            }}, 5000);
            function copyReferral() {{
              navigator.clipboard.writeText(document.getElementById("referral_link").value)
                .then(() => alert("Referral link copied to clipboard!"));
            }}
          </script>
        </head>
        <body>
          <h2>Your request has been approved!</h2>
          <p>Your remaining upload credit: {credit}{pro_notice}</p>
          <p><a class="btn" href="/confirm_upload?token={SECRET_TOKEN}">Confirm and Proceed with Uploads</a></p>
          <p>Share your referral link to earn extra credits:</p>
          <input type="text" id="referral_link" value="{referral_link}" readonly/>
          <button onclick="copyReferral()">Copy Referral Link</button>
          <hr/>
          <h3>Upgrade to Pro for Premium Features</h3>
          <p>
            <a class="btn" href="/go_pro?token={SECRET_TOKEN}">Go Pro – £10/month</a>
            <a class="btn" href="/go_pro?token={SECRET_TOKEN}">Go Pro – £55/year (Best Value)</a>
          </p>
          <p><a href="/logout?token={SECRET_TOKEN}" style="color:#f66;">Logout</a></p>
        </body>
        </html>
        """
        return html
    elif username in PENDING_REQUESTS:
        html = f"""
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Status: Pending Approval</title>
          <link rel="manifest" href="/static/manifest.json">
          <meta name="theme-color" content="#28a745">
          <style>
            body {{
              background: linear-gradient(135deg, #111, #333);
              color: #fff; text-align: center; font-family: Arial; padding: 50px;
            }}
            a {{ color: #0af; text-decoration: none; }}
            h2 {{ text-shadow: 0 0 5px #f0f; }}
          </style>
          <script>
            setTimeout(function(){{ window.location.href="/user_status?token={SECRET_TOKEN}"; }}, 5000);
          </script>
        </head>
        <body>
          <h2>Your request is pending admin approval.</h2>
          <p>Please wait while we process your request...</p>
          <p><a href="/user_status?token={SECRET_TOKEN}">Check Status</a></p>
          <p><a href="/?token={SECRET_TOKEN}">Back to Main Page</a></p>
        </body>
        </html>
        """
        return html
    else:
        return "<h3>No request found. Please submit your analysis form.</h3><a href='/analysis_form?token={}' >Go to Form</a>".format(SECRET_TOKEN)

# ========================
# CONFIRM UPLOADS & ADDITIONAL UPLOADS
# ========================
@app.route("/confirm_upload")
def confirm_upload():
    username = session.get("username")
    if not username or username not in APPROVED_REQUESTS:
        return redirect(url_for("user_status", token=SECRET_TOKEN))
    user_data = APPROVED_REQUESTS[username]
    videos = user_data.get("videos", [])
    credit = user_data.get("upload_credit", 0)
    uid = user_data.get("uid", "")
    video_list_html = "<ul>" + "".join(f"<li>{v}</li>" for v in videos) + "</ul>"
    html = f"""
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Confirm Uploads</title>
      <link rel="manifest" href="/static/manifest.json">
      <meta name="theme-color" content="#28a745">
      <style>
        body {{
          background: linear-gradient(135deg, #111, #333); color: #fff; text-align: center; font-family: Arial; padding: 50px;
        }}
        a, button {{
          padding: 10px 20px; background: #28a745; color: #fff; border: none; border-radius: 4px;
          text-decoration: none; margin: 10px; cursor: pointer; transition: box-shadow 0.3s;
        }}
        a:hover, button:hover {{ box-shadow: 0 0 10px #28a745; }}
        ul {{ list-style: none; padding: 0; }}
      </style>
    </head>
    <body>
      <h2>Confirm Your Uploaded Videos</h2>
      <p>Uploaded Videos:</p>
      {video_list_html}
      <p>Your remaining upload credit: {credit}</p>
      <p>If you wish to upload more (each additional video deducts one credit), do so below:</p>
      <form action="/upload_more?token={SECRET_TOKEN}" method="POST" enctype="multipart/form-data">
        <input type="file" name="upload_files" multiple accept=".mp4,.avi,.mov" required/><br/>
        <button type="submit">Upload Additional Videos</button>
      </form>
      <p>
        <a href="/play_video?uid={uid}&token={SECRET_TOKEN}">Proceed to Video Analysis</a>
      </p>
      <p><a href="/logout?token={SECRET_TOKEN}" style="background:#f66;">Logout</a></p>
    </body>
    </html>
    """
    return html

@app.route("/upload_more", methods=["POST"])
def upload_more():
    username = session.get("username")
    if not username or username not in APPROVED_REQUESTS:
        return redirect(url_for("user_status", token=SECRET_TOKEN))
    user_data = APPROVED_REQUESTS[username]
    current_credit = user_data.get("upload_credit", 0)
    files = request.files.getlist("upload_files")
    num_files = len(files)
    if num_files > current_credit and current_credit != -1:
        return f"<h3>You only have {current_credit} upload credits remaining.</h3><a href='/confirm_upload?token={SECRET_TOKEN}'>Go Back</a>"
    new_videos = []
    for f in files:
        if f.filename and allowed_file(f.filename):
            filename = f"{username}_{uuid.uuid4().hex}_{f.filename}"
            sp = os.path.join(UPLOAD_DIR, filename)
            f.save(sp)
            new_videos.append(filename)
        else:
            return "<h3>Invalid file type uploaded.</h3><a href='/confirm_upload?token={token}'>Go Back</a>".format(token=SECRET_TOKEN)
    user_data["videos"].extend(new_videos)
    if current_credit != -1:
        user_data["upload_credit"] = current_credit - num_files
    APPROVED_REQUESTS[username] = user_data
    add_log(USER_LOGS, f"User '{username}' uploaded {num_files} additional video(s).", "user")
    return redirect(url_for("confirm_upload", token=SECRET_TOKEN))

# ========================
# YOLO PIPELINE FOR VIDEO PROCESSING
# ========================
def bbox_overlap(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xx1 = max(x1, x2)
    yy1 = max(y1, y2)
    xx2 = min(x1+w1, x2+w2)
    yy2 = min(y1+h1, y2+h2)
    ow = xx2 - xx1
    oh = yy2 - yy1
    return ow*oh if (ow > 0 and oh > 0) else 0

def detect_jersey_number(region, target_number):
    config = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(region, config=config)
    return str(target_number) in text.strip()

def yoyov5_process_video(uid, video_path, jersey_colors, boots_colors, frames_done, total_frames, player_name, target_jersey):
    model = global_yolo_model
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open => {video_path}")
        return [], None
    detection_history = []
    processed_index = 0
    local_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    best_tackle_value = 0
    best_tackle_frame = None
    best_tackle_bbox = None
    fallback_area = 0
    fallback_frame = None
    fallback_bbox = None
    last_frame = None
    prev_center = None
    prev_bbox = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame.copy()
        cv2.putText(frame, f"#{target_jersey}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        GLOBAL_LIVE_FRAME[uid] = frame.copy()
        results = model(frame, conf=YOLO_CONF_THRESHOLD)
        person_boxes = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            if cls_id == 0:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                w = int(x2 - x1)
                h = int(y2 - y1)
                person_boxes.append((int(x1), int(y1), w, h))
        selected_box = None
        if person_boxes:
            for box in person_boxes:
                (x, y, w, h) = box
                region = frame[y:y+h, x:x+w]
                if detect_jersey_number(region, target_jersey):
                    selected_box = box
                    break
            if not selected_box:
                selected_box = max(person_boxes, key=lambda b: b[2]*b[3])
            (x, y, w, h) = selected_box
            bbox_center = (x + w/2, y + h/2)
            bbox_area = w * h
            collisions = 0
            velocity = 0.0
            if prev_bbox is not None:
                overlap = bbox_overlap(selected_box, prev_bbox)
                if overlap > COLLISION_THRESHOLDS[1]:
                    collisions = 1
                if overlap > COLLISION_THRESHOLDS[2]:
                    collisions = 2
            if prev_center is not None:
                dx = bbox_center[0] - prev_center[0]
                dy = bbox_center[1] - prev_center[1]
                velocity = abs(dx) + abs(dy)
            prev_bbox = selected_box
            prev_center = bbox_center
            frame_height = frame.shape[0]
            is_lower_third = (bbox_center[1] > frame_height * 0.66)
            tackles = collisions if (is_lower_third and velocity > 1.0 and collisions >= 1) else 0
            if tackles > 0 or collisions >= 2:
                ret_enc, buff = cv2.imencode('.jpg', frame)
                if ret_enc:
                    encoded = base64.b64encode(buff).decode('utf-8')
                    ev_type = "tackle" if tackles > 0 else "interception"
                    GLOBAL_EVENT_FRAMES.setdefault(uid, []).append((ev_type, encoded))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            bbox_roi = hsv[y:y+h, x:x+w]
            color_pixels = 0
            for (lower, upper) in (jersey_colors + boots_colors):
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(bbox_roi, lower_np, upper_np)
                color_pixels += cv2.countNonZero(mask)
            scan_count = color_pixels // 3000
            detection_history.append({
                "frame": processed_index,
                "velocity": float(velocity),
                "collisions": collisions,
                "scan_count": scan_count,
                "tackles": tackles,
                "bbox": selected_box
            })
            if tackles > best_tackle_value:
                best_tackle_value = tackles
                best_tackle_frame = frame.copy()
                best_tackle_bbox = selected_box
            if bbox_area > fallback_area:
                fallback_area = bbox_area
                fallback_frame = frame.copy()
                fallback_bbox = selected_box
        else:
            detection_history.append({
                "frame": processed_index,
                "velocity": 0.0,
                "collisions": 0,
                "scan_count": 0,
                "tackles": 0,
                "bbox": None
            })
        processed_index += 1
        frames_done[0] += 1
        GLOBAL_PROGRESS[uid] = int(frames_done[0] / total_frames * 100)
    cap.release()
    chosen_frame = best_tackle_frame if best_tackle_frame is not None and best_tackle_value > 0 else fallback_frame
    chosen_bbox = best_tackle_bbox if best_tackle_frame is not None and best_tackle_value > 0 else fallback_bbox
    tackle_image_path = None
    if chosen_frame is not None and chosen_bbox is not None:
        (cx, cy, cw, ch) = chosen_bbox
        snippet = chosen_frame[cy:cy+ch, cx:cx+cw]
        if snippet.size > 0:
            tackle_image_path = f"player_tackle_{uid}_{random.randint(1000,9999)}.jpg"
            cv2.imwrite(tackle_image_path, snippet)
    if tackle_image_path is None and last_frame is not None:
        tackle_image_path = f"player_tackle_{uid}_{random.randint(1000,9999)}.jpg"
        cv2.imwrite(tackle_image_path, last_frame)
    return detection_history, tackle_image_path

# ========================
# PLAY VIDEO ENDPOINT
# ========================
@app.route("/play_video")
def play_video():
    username = session.get("username")
    if not username or username not in APPROVED_REQUESTS:
        return redirect(url_for("user_status", token=SECRET_TOKEN))
    uid = request.args.get("uid", "")
    if not uid:
        return "No user ID provided."
    user_data = APPROVED_REQUESTS[username]
    name = user_data.get("player_name", "Defender")
    age = user_data.get("age", "26")
    jersey_number = user_data.get("jersey_number", "10")
    nationality = user_data.get("nationality", "Others")
    video_list = user_data.get("videos", [])
    add_log(ADMIN_LOGS, f"[VIDEO ANALYSIS] User '{username}' arrived on 'play_video' page with UID={uid}.", "admin")
    html = """
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Confirm Videos (UID={{uid}})</title>
      <link rel="manifest" href="/static/manifest.json">
      <meta name="theme-color" content="#28a745">
      <style>
        body { background: linear-gradient(135deg, #111, #333); color: #fff; text-align: center; font-family: Arial; padding: 20px; }
        .container { max-width: 800px; margin: 50px auto; background: rgba(0,0,0,0.7); padding: 20px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.5); }
        button { padding: 10px 20px; margin: 10px; cursor: pointer; background: #28a745; color: #fff; border: none; border-radius: 4px; transition: box-shadow 0.3s; }
        button:hover { box-shadow: 0 0 10px #28a745; }
        a { color: #0af; text-decoration: none; }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Videos to Analyze (UID={{uid}})</h1>
        <ul style="text-align: left;">
          {% for vinfo in video_list %}
          <li>{{ vinfo }}</li>
          {% endfor %}
        </ul>
        <p>If correct, proceed to analyze.</p>
        <a href="{{ url_for('run_pipeline', uid=uid, name=name, age=age, jersey_number=jersey_number, nationality=nationality, token=token) }}">
          <button>Analyze &amp; Generate Plan</button>
        </a>
        <a href="/confirm_upload?token={{token}}">
          <button>Back to Confirm Uploads</button>
        </a>
      </div>
    </body>
    </html>
    """
    return render_template_string(html, uid=uid, video_list=video_list, name=name, age=age, jersey_number=jersey_number, nationality=nationality, token=SECRET_TOKEN)

# ========================
# RUN PIPELINE ENDPOINT
# ========================
@app.route("/run_pipeline")
def run_pipeline():
    username = session.get("username")
    if not username or username not in APPROVED_REQUESTS:
        return redirect(url_for("user_status", token=SECRET_TOKEN))
    uid = request.args.get("uid", "")
    if not uid:
        return "No user ID provided; pipeline cannot run."
    done = request.args.get("done", "")
    if done == "true":
        results = GLOBAL_RESULTS.get(uid, None)
        if not results:
            return f"No results or pipeline not finished for uid={uid}"
        video_data = results.get("videos", [])
        name = results.get("name", "Defender")
        age = results.get("age", "26")
        nationality = results.get("nationality", "Others")
        contact = username
        if video_data:
            sample_current_stats = video_data[-1].get("current_stats", {})
            sample_predicted_stats = video_data[-1].get("predicted_stats", {})
        else:
            sample_current_stats = {}
            sample_predicted_stats = {}
        training_regimen = generate_training_regimen(age, sample_current_stats, sample_predicted_stats)
        drill_recs = {
            "tackling": "Practice 1v1 shadowing. Watch this video: [link]",
            "agility": "Try ladder drills 3x per week. PDF attached.",
            "stamina": "Incorporate interval training to boost endurance."
        }
        final_html = render_template_string("""
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Analysis Results (UID={{uid}})</title>
          <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1/dist/confetti.browser.min.js"></script>
          <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
          <link rel="manifest" href="/static/manifest.json">
          <meta name="theme-color" content="#28a745">
          <style>
            body { background: linear-gradient(135deg, #111, #333); color: #eee; font-family: Arial; margin: 0; padding: 0; }
            .header { background: rgba(0,0,0,0.7); padding: 20px; text-align: center; text-shadow: 0 0 5px #0f0; }
            .container { max-width: 900px; margin: 40px auto; background: rgba(25,25,25,0.9); padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.5); color: #fff; margin-bottom: 20px; }
            .attributes { display: flex; flex-wrap: wrap; justify-content: center; margin: 10px 0; }
            .attr-box { background: linear-gradient(135deg, #ff7e5f, #feb47b); border-radius: 6px; padding: 5px; margin: 5px; text-align: center; width: 120px; box-shadow: inset 0 0 10px #000; }
            .send-btn, .invite-btn, .go-pro-btn { padding: 10px 20px; background: #28a745; color: #fff; border: none; border-radius: 4px; cursor: pointer; margin: 10px; transition: box-shadow 0.3s; }
            .send-btn:hover { box-shadow: 0 0 10px #28a745; }
            .invite-btn { background: #0af; }
            .invite-btn:hover { box-shadow: 0 0 10px #0af; }
            .go-pro-btn { background: #ff9900; }
            .go-pro-btn:hover { box-shadow: 0 0 10px #ff9900; }
            .chatbox { margin-top: 20px; }
            .chatbox textarea { width: 80%; height: 60px; padding: 8px; border-radius: 4px; border: none; box-shadow: inset 0 0 5px #555; }
            .chatbox button { padding: 8px 16px; margin-left: 10px; background: #0af; border: none; border-radius: 4px; color: #fff; transition: box-shadow 0.3s; }
            .chatbox button:hover { box-shadow: 0 0 10px #0af; }
            #chartContainer { width: 80%; margin: 20px auto; }
            .regimen-box { background: #222; padding: 15px; border-radius: 8px; margin-top: 20px; text-align: left; white-space: pre-wrap; }
          </style>
          <script>
            function runConfetti(){
              var end = Date.now() + 5000;
              (function frame(){
                confetti({ particleCount: 3, angle: 60, spread: 55, origin: { x: 0 } });
                confetti({ particleCount: 3, angle: 120, spread: 55, origin: { x: 1 } });
                if(Date.now() < end) { requestAnimationFrame(frame); }
              })();
            }
            window.onload = function(){
              runConfetti();
              document.getElementById('celebration_audio').play();
              var ctx = document.getElementById('progressChart').getContext('2d');
              var chart = new Chart(ctx, {
                type: 'line',
                data: {
                  labels: ['Year 1', 'Year 2', 'Year 3'],
                  datasets: [{
                    label: 'Performance Progression',
                    data: [{{data1}}, {{data2}}, {{data3}}],
                    borderColor: '#0f0',
                    fill: false,
                  }]
                },
                options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }
              });
            }
            function sendCopy(){
              fetch('/send_report?uid={{uid}}&token={{token}}').then(r => r.text()).then(alert);
            }
            function inviteFriend() {
              fetch('/invite_friend?uid={{uid}}&token={{token}}').then(r => r.text()).then(alert);
            }
            function sendChat(){
              var msg = document.getElementById("chat_msg").value;
              fetch('/chat?uid={{uid}}&msg=' + encodeURIComponent(msg) + '&token={{token}}')
                .then(r => r.text()).then(alert);
            }
          </script>
        </head>
        <body>
          <audio id="celebration_audio" src="/serve_audio/celebration.mp3"></audio>
          <div class="header">
            <h2>Analysis Results (UID={{uid}}) - Pro Edition</h2>
          </div>
          {% for vinfo in video_data %}
          <div class="container">
            <h2>Video #{{ loop.index }}: {{ vinfo.video_filename }}</h2>
            <div style="text-align:center; margin-bottom:10px;">
              {% if vinfo.tackle_image_path %}
              <img src="/snippet/{{ vinfo.tackle_image_path }}?uid={{uid}}" style="width:250px; border-radius:8px;"/>
              {% endif %}
            </div>
            <h3 style="color:#ffeb3b;">Current Stats</h3>
            <div class="attributes">
              {% for key, value in vinfo.current_stats.items() %}
              {% set color = "#0f0" if value >= 80 else "#f00" if value < 50 else "#ff0" %}
              <div class="attr-box" style="color:{{color}};">
                <strong>{{ key }}</strong><br/>
                <span>{{ "%.2f"|format(value) }}</span>
              </div>
              {% endfor %}
            </div>
            <h3 style="color:#8bc34a;">Predicted Stats</h3>
            <div class="attributes">
              {% for key, value in vinfo.predicted_stats.items() %}
              {% set color = "#0f0" if value >= 80 else "#f00" if value < 50 else "#ff0" %}
              <div class="attr-box" style="color:{{color}};">
                <strong>{{ key }}</strong><br/>
                <span>{{ "%.2f"|format(value) }}</span>
              </div>
              {% endfor %}
            </div>
            <h3>Training Recommendations</h3>
            <ul>
              {% for rec_key, rec_text in drill_recs.items() %}
              <li><strong>{{ rec_key.capitalize() }}:</strong> {{ rec_text }}</li>
              {% endfor %}
            </ul>
            <div class="regimen-box">
              <h3>Your 3-Year Training Regimen</h3>
              <pre>{{ training_regimen }}</pre>
            </div>
          </div>
          {% endfor %}
          <div class="container" style="text-align:center;">
            <div id="chartContainer">
              <canvas id="progressChart"></canvas>
            </div>
            <button class="send-btn" onclick="sendCopy()">Get a Copy</button>
            <button class="invite-btn" onclick="inviteFriend()">Invite a Friend (+5 uploads)</button>
            <button class="go-pro-btn" onclick="window.location.href='/go_pro?token={{token}}'">Go Pro Now!</button>
            <p style="font-size:0.9em; color:#aaa;">User: {{contact}}</p>
            <p><a href="/logout?token={{token}}" style="color:#f66;">Logout</a></p>
            <div class="chatbox">
              <h4>Chat with Support</h4>
              <textarea id="chat_msg" placeholder="Type your message..."></textarea>
              <button onclick="sendChat()">Send</button>
            </div>
          </div>
          <div id="liveChatBox" style="position: fixed; bottom: 100px; right: 20px; background: #333; color: #fff; padding:10px; border-radius: 8px; display: none;">
            <div id="chatMessages" style="max-height:150px; overflow-y:auto; text-align:left;"></div>
            <input type="text" id="userChatInput" style="width:80%; padding:5px;" placeholder="Type your message..."/>
            <button onclick="sendUserChat()">Send</button>
            <div id="ratingPrompt" style="display:none;">
              <p>Please rate the admin (1-5 stars):</p>
              <input type="number" id="adminRating" min="1" max="5"/>
              <button onclick="submitRating()">Submit Rating</button>
            </div>
          </div>
          <script>
          function pollLiveChat(){
            fetch('/user_live_chat?token={{token}}')
              .then(r => r.json())
              .then(data => {
                if(data.active || data.ended){
                  document.getElementById("liveChatBox").style.display = "block";
                  var messagesDiv = document.getElementById("chatMessages");
                  messagesDiv.innerHTML = "";
                  if(data.messages){
                    data.messages.forEach(function(msg){
                      messagesDiv.innerHTML += "<p><strong>" + msg[0] + ":</strong> " + msg[1] + "</p>";
                    });
                  }
                  if(data.ended){ document.getElementById("ratingPrompt").style.display = "block"; }
                } else { document.getElementById("liveChatBox").style.display = "none"; }
              });
          }
          function sendUserChat(){
            var input = document.getElementById("userChatInput");
            var message = input.value;
            fetch('/chat?token={{token}}&uid={{uid}}&msg=' + encodeURIComponent(message))
              .then(r => r.text()).then(alert);
            input.value = "";
          }
          function submitRating(){
            var rating = document.getElementById("adminRating").value;
            fetch('/user_rate_admin?token={{token}}', {
              method: 'POST',
              headers: {'Content-Type': 'application/x-www-form-urlencoded'},
              body: 'rating=' + rating
            }).then(r => r.text()).then(function(response){
              alert(response);
              document.getElementById("ratingPrompt").style.display = "none";
            });
          }
          setInterval(pollLiveChat, 5000);
          </script>
        </body>
        </html>
        """, uid=uid, video_data=video_data, name=name, age=age, nationality=nationality, contact=contact, drill_recs=drill_recs,
            training_regimen=training_regimen, token=SECRET_TOKEN,
            data1=random.uniform(50,90), data2=random.uniform(50,90), data3=random.uniform(50,90))
        add_log(USER_LOGS, f"User '{username}' processing complete for uid={uid}.", "user")
        try:
            user_msg = Message("Your GODT Analysis Report",
                               recipients=[APPROVED_REQUESTS[username].get("email", f"{username}@example.com")])
            user_msg.html = final_html
            mail.send(user_msg)
        except Exception as e:
            print("[ERROR sending email]", e)
        return final_html
    name = request.args.get("name", "Defender")
    age = request.args.get("age", "26")
    jersey_number = request.args.get("jersey_number", "10")
    nationality = request.args.get("nationality", "Others")
    user_data = APPROVED_REQUESTS[username]
    video_list = user_data.get("videos", [])
    player_name = user_data.get("player_name", "Defender")
    target_jersey = user_data.get("jersey_number", "10")
    def pipeline_thread():
        total_frames = 0
        for v in video_list:
            path = os.path.join(UPLOAD_DIR, v)
            cap = cv2.VideoCapture(path)
            local_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            total_frames += local_frames
            cap.release()
        frames_done = [0]
        GLOBAL_EVENT_FRAMES[uid] = []
        video_results = []
        for v in video_list:
            path = os.path.join(UPLOAD_DIR, v)
            detection_history, tackle_image_path = yoyov5_process_video(
                uid, path, [], [], frames_done, total_frames, player_name, target_jersey
            )
            cstats = { attr: random.uniform(*STAT_RANGES.get(attr, (30,80))) for attr in ATTRIBUTES }
            pstats = { attr: max(0, min(100, cstats[attr] + random.uniform(-5,5))) for attr in ATTRIBUTES }
            video_results.append({
                "video_filename": v,
                "tackle_image_path": tackle_image_path,
                "current_stats": cstats,
                "predicted_stats": pstats
            })
        GLOBAL_RESULTS[uid] = { "videos": video_results, "name": name, "age": age, "nationality": nationality }
        GLOBAL_PROGRESS[uid] = 100
        session_details = (f"Session for user '{username}' (UID: {uid}) processed {len(video_list)} videos at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        add_log([], session_details, "session")
        # Save current session stats to memory
        store_player_data(username, video_results[-1]["current_stats"])
    t = threading.Thread(target=pipeline_thread)
    t.start()
    add_log(ADMIN_LOGS, f"[VIDEO ANALYSIS] Pipeline started for user '{username}' with UID={uid}.", "admin")
    wait_html = """
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Processing Videos... (UID={{uid}})</title>
      <link rel="manifest" href="/static/manifest.json">
      <meta name="theme-color" content="#28a745">
      <style>
        body { margin: 0; padding: 0; font-family: Arial; color: #fff; text-align: center; background: linear-gradient(135deg, #111, #333); overflow: hidden; }
        .overlay { background: rgba(0,0,0,0.7); width: 100%; height: 100vh; position: absolute; top: 0; left: 0; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 20px; }
        .progress-bar { width: 50%; height: 20px; background: #444; border-radius: 10px; overflow: hidden; margin-bottom: 5px; }
        .progress-fill { height: 100%; width: 0%; background: #28a745; transition: width 0.3s; }
        #progress-text { font-weight: bold; margin-bottom: 20px; }
        .speech-box { background: rgba(0,0,0,0.3); padding: 15px; border-radius: 6px; font-size: 1.1em; min-height: 80px; margin-bottom: 20px; max-width: 600px; }
        .lines-box { background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px; max-width: 600px; font-size: 1.2em; }
        h1 { text-shadow: 0 0 5px #0f0; }
        #gameCanvas { position: absolute; top: 10%; left: 10%; border: 2px solid #fff; display: none; background: #000; }
        #gameBtn, #closeGameBtn { position: fixed; bottom: 20px; padding: 10px 20px; background: #ff9900; border: none; border-radius: 8px; color: #fff; font-size: 16px; cursor: pointer; z-index: 20; }
        #closeGameBtn { display: none; bottom: 60px; }
        #playAudioBtn { padding: 10px 20px; background: #0af; border: none; border-radius: 4px; color: #fff; cursor: pointer; margin-top: 20px; display: none; }
      </style>
      <script>
        var speeches = {{ speeches|tojson }};
        var lines = {{ lines|tojson }};
        var speechIdx = 0, lineIdx = 0;
        function rotateSpeech(){ document.getElementById("speech").innerHTML = speeches[speechIdx]; speechIdx = (speechIdx + 1) % speeches.length; }
        function rotateLines(){ document.getElementById("lines").innerHTML = lines[lineIdx]; lineIdx = (lineIdx + 1) % lines.length; }
        function updateProgress(){
          let uid = '{{uid}}';
          fetch('/progress?uid=' + uid)
            .then(r => r.json())
            .then(data => {
              document.getElementById('fill').style.width = data.progress + "%";
              document.getElementById('progress-text').textContent = data.progress + "%";
              if(data.progress < 100){ setTimeout(updateProgress, 500); }
              else { window.location.href = window.location.href + "&done=true"; }
            });
        }
        function updateLivePreview(){ let uid = '{{uid}}'; document.getElementById('live_preview').src = '/live_frame?uid=' + uid + '&' + new Date().getTime(); }
        function updateEventPreview(){
          let uid = '{{uid}}';
          fetch('/event_frame?uid=' + uid)
            .then(r => r.json())
            .then(data => { if(data.image){ document.getElementById('event_preview').src = "data:image/jpeg;base64," + data.image; } });
        }
        function setupBackgroundMusic(){
          var bgAudio = document.getElementById('bg_audio');
          bgAudio.onended = function(){ bgAudio.src = bgAudio.src.indexOf("sample.mp3") !== -1 ? "/serve_audio/sample1.mp3" : "/serve_audio/sample.mp3"; bgAudio.play(); }
        }
        var gameInterval, gameIdleTimeout;
        var gameCanvas, ctx;
        var ball = { x: 50, y: 50, radius: 15, vx: 2, vy: 2 };
        var keeper = { x: 200, y: 50, width: 50, height: 80, vx: 1 };
        var defender = { x: 0, y: 0, width: 40, height: 60, active: false };
        var goal = { side: Math.random() < 0.5 ? 'left' : 'right', width: 20, height: 100 };
        var score = { user: 0, ai: 0 };
        var goalVisible = false, roundTimer;
        function startGame(){
          document.getElementById('gameCanvas').style.display = "block";
          document.getElementById('closeGameBtn').style.display = "block";
          resetRound();
          gameInterval = setInterval(updateGame, 30);
          resetIdleTimer();
        }
        function minimizeGame(){ document.getElementById('gameCanvas').style.display = "none"; document.getElementById('closeGameBtn').style.display = "none"; clearInterval(gameInterval); clearTimeout(roundTimer); }
        function resetRound(){
          goal.side = Math.random() < 0.5 ? 'left' : 'right';
          goalVisible = true;
          defender.active = (Math.random() < 0.3);
          if(defender.active){ defender.x = Math.random() * (gameCanvas.width - defender.width); defender.y = Math.random() * (gameCanvas.height - defender.height); }
          roundTimer = setTimeout(function(){ goalVisible = false; score.ai += 1; resetRound(); }, 8000);
        }
        function updateGame(){
          ball.x += ball.vx; ball.y += ball.vy;
          if(ball.x - ball.radius < 0 || ball.x + ball.radius > gameCanvas.width) ball.vx = -ball.vx;
          if(ball.y - ball.radius < 0 || ball.y + ball.radius > gameCanvas.height) ball.vy = -ball.vy;
          keeper.x += keeper.vx;
          if(keeper.x < 0 || keeper.x + keeper.width > gameCanvas.width){ keeper.vx = -keeper.vx; }
          if(defender.active){ defender.x += (Math.random() - 0.5) * 4; defender.y += (Math.random() - 0.5) * 4; defender.x = Math.max(0, Math.min(gameCanvas.width - defender.width, defender.x)); defender.y = Math.max(0, Math.min(gameCanvas.height - defender.height, defender.y)); if(ball.x + ball.radius > defender.x && ball.x - ball.radius < defender.x + defender.width && ball.y + ball.radius > defender.y && ball.y - ball.radius < defender.y + defender.height){ ball.vx = -ball.vx; ball.vy = -ball.vy; defender.active = false; } }
          if(goalVisible){
            if(goal.side === 'left' && ball.x - ball.radius <= goal.width){
              if(ball.y >= (gameCanvas.height - goal.height) / 2 && ball.y <= (gameCanvas.height + goal.height) / 2){ goalScored(); }
            }
            if(goal.side === 'right' && ball.x + ball.radius >= gameCanvas.width - goal.width){
              if(ball.y >= (gameCanvas.height - goal.height) / 2 && ball.y <= (gameCanvas.height + goal.height) / 2){ goalScored(); }
            }
          }
          drawGame();
        }
        function drawGame(){
          ctx.clearRect(0, 0, gameCanvas.width, gameCanvas.height);
          ctx.beginPath();
          ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
          ctx.fillStyle = "#fff"; ctx.fill(); ctx.closePath();
          var keeperImg = new Image(); keeperImg.src = "/serve_keeper_image?token={{token}}"; ctx.drawImage(keeperImg, keeper.x, keeper.y, keeper.width, keeper.height);
          if(defender.active){ ctx.fillStyle = "#ffa500"; ctx.fillRect(defender.x, defender.y, defender.width, defender.height); }
          if(goalVisible){
            ctx.fillStyle = "#f00";
            if(goal.side === 'left'){ ctx.fillRect(0, (gameCanvas.height - goal.height) / 2, goal.width, goal.height); }
            else { ctx.fillRect(gameCanvas.width - goal.width, (gameCanvas.height - goal.height) / 2, goal.width, goal.height); }
          }
          ctx.font = "16px Arial"; ctx.fillStyle = "#fff"; ctx.fillText("Score: You " + score.user + " - AI " + score.ai, 10, 20);
        }
        function goalScored(){ clearTimeout(roundTimer); goalVisible = false; score.user += 1; alert("Goal! You scored vs AItecky!"); resetRound(); }
        function canvasClick(e){
          var rect = gameCanvas.getBoundingClientRect();
          var clickX = e.clientX - rect.left, clickY = e.clientY - rect.top;
          var dist = Math.sqrt((clickX - ball.x)**2 + (clickY - ball.y)**2);
          if(dist < ball.radius + 10){ ball.vx = (Math.random() * 4) - 2; ball.vy = (Math.random() * 4) - 2; }
        }
        function resetIdleTimer(){ clearTimeout(gameIdleTimeout); gameIdleTimeout = setTimeout(function(){ minimizeGame(); }, 30 * 60 * 1000); }
        window.onload = function(){
          setupBackgroundMusic();
          var audioEl = document.getElementById('bg_audio');
          var playBtn = document.getElementById('playAudioBtn');
          audioEl.play().catch(function(){ playBtn.style.display = "block"; });
          playBtn.onclick = function(){ audioEl.play(); playBtn.style.display = "none"; };
          rotateSpeech(); rotateLines();
          setInterval(rotateSpeech, 15000); setInterval(rotateLines, 7000);
          updateProgress(); setInterval(updateLivePreview, 1000); setInterval(updateEventPreview, 3000);
          gameCanvas = document.getElementById("gameCanvas"); ctx = gameCanvas.getContext("2d");
          gameCanvas.width = 400; gameCanvas.height = 300; gameCanvas.addEventListener("click", canvasClick);
        }
      </script>
    </head>
    <body>
      <audio id="bg_audio" src="/serve_audio/sample.mp3" playsinline></audio>
      <button id="playAudioBtn">Tap to Play Audio</button>
      <div class="overlay">
        <div class="speech-box" id="speech"></div>
        <h1 style="margin-bottom:10px;">Processing Your Videos... (UID={{uid}})</h1>
        <div class="progress-bar"><div id="fill" class="progress-fill"></div></div>
        <div id="progress-text">0%</div>
        <div class="lines-box" id="lines"></div>
        <div style="margin-top:20px;">
          <img id="live_preview" alt="Live Frame" style="width:200px; border:3px solid #28a745;"/>
          <img id="event_preview" alt="Key Event Frame" style="width:200px; border:3px solid #ff6666; margin-left:10px;"/>
        </div>
      </div>
      <button id="gameBtn" onclick="startGame()">Play Waiting Game</button>
      <button id="closeGameBtn" onclick="minimizeGame()">Minimize Game</button>
      <canvas id="gameCanvas"></canvas>
      <div id="liveChatBox" style="position: fixed; bottom: 100px; right: 20px; background: #333; color: #fff; padding:10px; border-radius: 8px; display: none;">
        <div id="chatMessages" style="max-height:150px; overflow-y:auto; text-align:left;"></div>
        <input type="text" id="userChatInput" style="width:80%; padding:5px;" placeholder="Type your message..."/>
        <button onclick="sendUserChat()">Send</button>
        <div id="ratingPrompt" style="display:none;">
          <p>Please rate the admin (1-5 stars):</p>
          <input type="number" id="adminRating" min="1" max="5"/>
          <button onclick="submitRating()">Submit Rating</button>
        </div>
      </div>
      <script>
      function pollLiveChat(){
        fetch('/user_live_chat?token={{token}}')
          .then(r => r.json())
          .then(data => {
            if(data.active || data.ended){
              document.getElementById("liveChatBox").style.display = "block";
              var messagesDiv = document.getElementById("chatMessages");
              messagesDiv.innerHTML = "";
              if(data.messages){
                data.messages.forEach(function(msg){
                  messagesDiv.innerHTML += "<p><strong>" + msg[0] + ":</strong> " + msg[1] + "</p>";
                });
              }
              if(data.ended){ document.getElementById("ratingPrompt").style.display = "block"; }
            } else { document.getElementById("liveChatBox").style.display = "none"; }
          });
      }
      function sendUserChat(){
        var input = document.getElementById("userChatInput");
        var message = input.value;
        fetch('/chat?token={{token}}&uid={{uid}}&msg=' + encodeURIComponent(message))
          .then(r => r.text()).then(alert);
        input.value = "";
      }
      function submitRating(){
        var rating = document.getElementById("adminRating").value;
        fetch('/user_rate_admin?token={{token}}', {
          method: 'POST',
          headers: {'Content-Type': 'application/x-www-form-urlencoded'},
          body: 'rating=' + rating
        }).then(r => r.text()).then(function(response){
          alert(response);
          document.getElementById("ratingPrompt").style.display = "none";
        });
      }
      setInterval(pollLiveChat, 5000);
      </script>
    </body>
    </html>
    """
    return render_template_string(wait_html, speeches=LONG_SPEECHES, lines=TRIVIA_LINES, uid=uid, token=SECRET_TOKEN)

# ========================
# SNIPPET, PROGRESS, LIVE & EVENT FRAME ENDPOINTS
# ========================
@app.route("/snippet/<filename>")
def serve_snippet(filename):
    return send_from_directory(os.getcwd(), filename)

@app.route("/progress")
def progress():
    uid = request.args.get("uid", "")
    val = GLOBAL_PROGRESS.get(uid, 0)
    return jsonify({"progress": val})

@app.route("/live_frame")
def live_frame():
    uid = request.args.get("uid", "")
    frame = GLOBAL_LIVE_FRAME.get(uid, None)
    if frame is None:
        return "", 204
    ret, buf = cv2.imencode('.jpg', frame)
    if not ret:
        return "", 500
    return app.response_class(buf.tobytes(), mimetype="image/jpeg")

@app.route("/event_frame")
def event_frame():
    uid = request.args.get("uid", "")
    evlist = GLOBAL_EVENT_FRAMES.get(uid, [])
    if evlist:
        e = random.choice(evlist)
        return jsonify({"event_type": e[0], "image": e[1]})
    return jsonify({"event_type": "none", "image": ""})

# ========================
# EMAIL REPORT ENDPOINT
# ========================
@app.route("/send_report")
def send_report():
    uid = request.args.get("uid", "")
    if not uid or uid not in GLOBAL_RESULTS:
        return "No user or results found."
    res = GLOBAL_RESULTS.get(uid)
    video_data = res.get("videos", [])
    name = res.get("name", "Defender")
    age = res.get("age", "26")
    nationality = res.get("nationality", "Others")
    username = session.get("username", "unknown_user")
    final_report_html = render_template_string("""
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>GODT Analysis Report</title>
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      <link rel="manifest" href="/static/manifest.json">
      <meta name="theme-color" content="#28a745">
      <style>
        body { background: #222; color: #fff; font-family: Arial; text-align: center; padding: 20px; }
        #chartContainer { width: 80%; margin: 20px auto; }
      </style>
      <script>
        window.onload = function(){
          var ctx = document.getElementById('progressChart').getContext('2d');
          var chart = new Chart(ctx, {
            type: 'line',
            data: {
              labels: ['Year 1', 'Year 2', 'Year 3'],
              datasets: [{
                label: 'Performance Progression',
                data: [{{data1}}, {{data2}}, {{data3}}],
                borderColor: '#0f0',
                fill: false,
              }]
            },
            options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }
          });
        }
      </script>
    </head>
    <body>
      <h1>GODT Analysis Report</h1>
      <p>Name: {{name}}<br/>Age: {{age}}<br/>Nationality: {{nationality}}</p>
      <p>Videos processed: {{video_count}}</p>
      <p>Generated for user: {{username}}</p>
      <div id="chartContainer">
        <canvas id="progressChart"></canvas>
      </div>
      <p>Check your analysis on our website!</p>
    </body>
    </html>
    """, name=name, age=age, nationality=nationality, video_count=len(video_data), username=username,
         data1=random.uniform(50, 90), data2=random.uniform(50, 90), data3=random.uniform(50, 90))
    try:
        user_msg = Message("Your GODT Analysis Report",
                            recipients=[APPROVED_REQUESTS[username].get("email", f"{username}@example.com")])
        user_msg.html = final_report_html
        mail.send(user_msg)
    except Exception as e:
        print("[ERROR sending email]", e)
    return f"Report sent to you (User: {username})."

# ========================
# INVITE FRIEND ENDPOINT
# ========================
@app.route("/invite_friend")
def invite_friend():
    username = session.get("username")
    if username and username in APPROVED_REQUESTS:
        APPROVED_REQUESTS[username]["upload_credit"] += 5
        add_log(USER_LOGS, f"User '{username}' invited a friend and earned 5 extra upload credits.", "user")
        return f"5 upload credits added! New credit: {APPROVED_REQUESTS[username]['upload_credit']}"
    return "User not found or not approved."

# ========================
# CHAT ENDPOINT (For user messages)
# ========================
@app.route("/chat")
def chat():
    uid = request.args.get("uid", "")
    msg = request.args.get("msg", "")
    if uid:
        from_user = session.get("username", "unknown")
        CHAT_MESSAGES.setdefault(uid, []).append((from_user, msg))
        if from_user in LIVE_CHAT and LIVE_CHAT[from_user]["active"]:
            LIVE_CHAT[from_user]["messages"].append((from_user, msg))
        add_log(USER_LOGS, f"Chat message from user '{from_user}': {msg}", "user")
        return "Message sent!"
    return "No UID provided."

# ========================
# RETRAIN MODEL ENDPOINT
# ========================
@app.route("/retrain", methods=["POST"])
def retrain():
    new_data = request.json
    if not new_data:
        return "No JSON data", 400
    csv_path = "player_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        cols = ["collisions", "velocity", "color_scans",
                "final_speed", "final_strength", "final_awareness", "final_tackling",
                "final_agility", "final_stamina", "final_positioning", "final_interceptions",
                "final_leadership", "final_vision"]
        df = pd.DataFrame(columns=cols)
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    subprocess.run(["python", "train_model.py"], check=False)
    add_log(ADMIN_LOGS, "Model re-trained with new data.", "admin")
    return "Model re-trained with new data!", 200

# ========================
# PRO PLAN & PAYMENT ENDPOINTS
# ========================
@app.route("/go_pro")
def go_pro():
    username = session.get("username", "unknown")
    add_log(ADMIN_LOGS, f"[PRO ALERT] User '{username}' has clicked Go Pro (upgrade attempt).", "admin")
    if username in APPROVED_REQUESTS:
        APPROVED_REQUESTS[username]["account_type"] = "pro"
        APPROVED_REQUESTS[username]["upload_credit"] = -1  # Unlimited uploads
        add_log(USER_LOGS, f"User '{username}' upgraded to Pro.", "user")
        add_log(ADMIN_LOGS, f"[PRO ALERT] User '{username}' is now a Pro user.", "admin")
    return redirect(url_for("go_pro_celebration", token=SECRET_TOKEN))

@app.route("/go_pro_celebration")
def go_pro_celebration():
    html = """
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Go Pro - Welcome, King!</title>
      <link rel="manifest" href="/static/manifest.json">
      <meta name="theme-color" content="#28a745">
      <style>
        body { background: radial-gradient(circle, #111, #333); color: #fff; font-family: Arial; text-align: center; padding: 50px; }
        .celebrate { font-size: 2em; color: #ff0; text-shadow: 0 0 10px #ff0; margin-top: 20px; }
        .btn { display: inline-block; padding: 15px 30px; margin: 20px; background: #28a745; color: #fff; text-decoration: none; border-radius: 8px; transition: box-shadow 0.3s; }
        .btn:hover { box-shadow: 0 0 12px rgba(40,167,69,0.7); }
      </style>
      <script>
        window.onload = function(){
          document.getElementById('pro_audio').play();
          alert("Alert to all admins: A user has upgraded to Pro!");
          setTimeout(function(){ window.location.href="/pro_dashboard?token={{token}}"; }, 5000);
        }
      </script>
    </head>
    <body>
      <audio id="pro_audio" src="/serve_audio/celebration.mp3"></audio>
      <h1>Welcome, King!</h1>
      <div class="celebrate">Max Celebration! Enjoy your premium perks, including scout sharing & unlimited uploads.</div>
      <p><a class="btn" href="/pro_dashboard?token={{token}}">Go to Your Pro Dashboard</a></p>
      <p><a class="btn" href="/?token={{token}}">Back to Home</a></p>
    </body>
    </html>
    """
    return render_template_string(html, token=SECRET_TOKEN)

@app.route("/process_payment")
def process_payment():
    plan = request.args.get("plan", "monthly")
    username = session.get("username", "unknown")
    if username in APPROVED_REQUESTS:
        APPROVED_REQUESTS[username]["account_type"] = "pro"
        APPROVED_REQUESTS[username]["upload_credit"] = -1
        add_log(USER_LOGS, f"User '{username}' upgraded to Pro ({plan}).", "user")
        add_log(ADMIN_LOGS, f"[PRO ALERT] User '{username}' is now a Pro user ({plan} plan).", "admin")
    return redirect(url_for("go_pro_celebration", token=SECRET_TOKEN))

# ========================
# PRO DASHBOARD (For Pro Users)
# ========================
@app.route("/pro_dashboard")
def pro_dashboard():
    username = session.get("username")
    if not username or username not in APPROVED_REQUESTS or APPROVED_REQUESTS[username].get("account_type") != "pro":
        return redirect(url_for("user_status", token=SECRET_TOKEN))
    profile_pic = APPROVED_REQUESTS[username].get("profile_pic")
    user_results = None
    for uid, res in GLOBAL_RESULTS.items():
        if res.get("name", "").lower() == APPROVED_REQUESTS[username].get("player_name", "").lower():
            user_results = res
            break
    if user_results and user_results.get("videos"):
        last_video = user_results["videos"][-1]
        profile_pic = last_video.get("tackle_image_path", profile_pic)
    html = """
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Pro Dashboard - GODT</title>
      <link rel="manifest" href="/static/manifest.json">
      <meta name="theme-color" content="#28a745">
      <style>
        body { background: linear-gradient(135deg, #222, #444); color: #fff; font-family: Arial, sans-serif; padding: 20px; }
        .profile { text-align: center; margin-bottom: 30px; }
        .profile img { width: 120px; height: 120px; border-radius: 50%; object-fit: cover; border: 4px solid #28a745; }
        .edit-btn { padding: 8px 16px; background: #0af; border: none; border-radius: 4px; color: #fff; cursor: pointer; margin-top: 10px; }
        .dashboard-card { background: #333; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .dashboard-card h3 { margin-top: 0; }
        .btn { display: inline-block; padding: 10px 20px; background: #28a745; color: #fff; border-radius: 4px; text-decoration: none; margin: 10px 5px; }
      </style>
    </head>
    <body>
      <div class="profile">
        <h1>Welcome, Pro {{username}}!</h1>
        {% if profile_pic %}
          <img src="/snippet/{{profile_pic}}" alt="Profile Picture"/>
        {% else %}
          <img src="/serve_bg_image" alt="Default Profile Picture"/>
        {% endif %}
        <br/>
        <button class="edit-btn" onclick="window.location.href='/upload_profile_pic?token={{token}}'">Edit Profile Picture</button>
      </div>
      <div class="dashboard-card">
        <h3>Your Latest Analysis</h3>
        <p>View your most recent analysis report and training recommendations.</p>
        <a class="btn" href="/run_pipeline?done=true&token={{token}}">View Report</a>
      </div>
      <div class="dashboard-card">
        <h3>Upload New Videos</h3>
        <p>You have unlimited uploads as a Pro user.</p>
        <a class="btn" href="/confirm_upload?token={{token}}">Upload Videos</a>
      </div>
      <div class="dashboard-card">
        <h3>Live Chat with Admin</h3>
        <p>Connect with support anytime.</p>
        <a class="btn" href="/user_status?token={{token}}">Open Chat</a>
      </div>
      <p><a class="btn" href="/logout?token={{token}}">Logout</a></p>
    </body>
    </html>
    """
    return render_template_string(html, username=username, profile_pic=profile_pic, token=SECRET_TOKEN)

# ========================
# PROFILE PICTURE UPLOAD (For Pro Users)
# ========================
@app.route("/upload_profile_pic", methods=["GET", "POST"])
def upload_profile_pic():
    username = session.get("username")
    if not username or username not in APPROVED_REQUESTS:
        return redirect(url_for("user_status", token=SECRET_TOKEN))
    if request.method == "POST":
        file = request.files.get("profile_pic")
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = f"{username}_profile_{uuid.uuid4().hex}.jpg"
            path = os.path.join(os.getcwd(), filename)
            file.save(path)
            APPROVED_REQUESTS[username]["profile_pic"] = filename
            add_log(USER_LOGS, f"User '{username}' updated their profile picture.", "user")
            return redirect(url_for("pro_dashboard", token=SECRET_TOKEN))
        else:
            return "Invalid file type."
    html = """
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Upload Profile Picture</title>
      <style>
        body { background: #222; color: #fff; text-align: center; font-family: Arial; padding: 50px; }
        .upload-box { background: #333; padding: 20px; border-radius: 8px; display: inline-block; }
        input { padding: 10px; margin: 10px; }
        button { padding: 10px 20px; background: #0af; border: none; border-radius: 4px; color: #fff; cursor: pointer; }
      </style>
    </head>
    <body>
      <div class="upload-box">
        <h2>Upload Your New Profile Picture</h2>
        <form method="POST" enctype="multipart/form-data">
          <input type="file" name="profile_pic" accept=".png,.jpg,.jpeg" required/><br/>
          <button type="submit">Upload</button>
        </form>
        <p><a href="/pro_dashboard?token={{token}}">Back to Pro Dashboard</a></p>
      </div>
    </body>
    </html>
    """
    return render_template_string(html, token=SECRET_TOKEN)

# ========================
# USER LOGIN (For Returning Users)
# ========================
@app.route("/user_login", methods=["GET", "POST"])
def user_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        if username in APPROVED_REQUESTS:
            session["username"] = username
            add_log(USER_LOGS, f"User '{username}' logged in (returning).", "user")
            return redirect(url_for("user_status", token=SECRET_TOKEN))
        else:
            return "<h3>Username not found. Please register via the analysis form.</h3><a href='/analysis_form?token={token}'>Register</a>".format(token=SECRET_TOKEN)
    html = """
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>User Login</title>
      <style>
        body { background: #111; color: #fff; text-align: center; font-family: Arial; padding: 50px; }
        .login-box { background: #333; padding: 20px; border-radius: 8px; display: inline-block; }
        input { padding: 10px; margin: 10px; width: 80%; }
        button { padding: 10px 20px; background: #28a745; border: none; border-radius: 4px; color: #fff; cursor: pointer; }
      </style>
    </head>
    <body>
      <div class="login-box">
        <h2>Returning User Login</h2>
        <form method="POST">
          <input type="text" name="username" placeholder="Enter your username" required/><br/>
          <button type="submit">Login</button>
        </form>
        <p><a href="/?token={{token}}" style="color:#0af;">Back to Main Page</a></p>
      </div>
    </body>
    </html>
    """
    return render_template_string(html, token=SECRET_TOKEN)

# ========================
# LOGOUT ENDPOINT
# ========================
@app.route("/logout")
def logout():
    uname = session.get("username")
    session_details = f"Session for user '{uname}' ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Final details: {APPROVED_REQUESTS.get(uname, PENDING_REQUESTS.get(uname, {}))}"
    add_log([], session_details, "session")
    if uname in PENDING_REQUESTS:
        PENDING_REQUESTS.pop(uname)
    if uname in APPROVED_REQUESTS:
        # For persistence, approved users should be stored externally.
        pass
    session.clear()
    add_log(USER_LOGS, f"User '{uname}' logged out.", "user")
    return redirect(url_for("main_page", token=SECRET_TOKEN))

# ========================
# CONVERSATIONAL COACHING (Already defined above as /coach_chat)
# ========================

# ========================
# RUN THE SERVER
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
