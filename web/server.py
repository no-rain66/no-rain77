import eventlet
eventlet.monkey_patch()  # 必须置顶

import cv2
import base64
import socket
import numpy as np
import webbrowser
import os
import sys
import json
import requests
import shutil
import threading
import platform
import time
import subprocess  
import re  # 🚀 用于提取文件名里的精确时间戳
from urllib.parse import quote
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

RK_IP = "192.168.0.232"
RK_PORT = 8888
RK_CMD_PORT = 8889
UDP_PORT = 9999
ASSETS_RECV_PORT = 9995  

latest_data = {
    "car_world_x": 0.0, "car_world_y": 0.0, "car_yaw": 0.0,
    "obstacle_in_1m": False, "recorded_path": [], "permanent_obstacles": [],
    "semantic_markers": [], "ai_running": False, "ai_counts": [0] * 7,
    "speed": 0.20, "turn_speed": 1.2,
    "min_dist": 0.15, "max_dist": 8.0,
    "nav_running": False, "detection_side": "right", "show_lidar_points": True,
    "battery_voltage": 12.0,
    "f1_soil_data": None,
    "spray_master_switch": False,
    "route_status": {}
}

map_clear_guard_until = 0.0

PATH_APPEND_MIN_DIST = 0.005   
PATH_MAX_POINTS = 12000        
PATH_JUMP_BREAK_DIST = 3.00    
PATH_FORCE_APPEND_INTERVAL = 0.8  

last_path_point = None
last_path_append_time = 0.0


def _sanitize_xy(x, y):
    try:
        return round(float(x), 4), round(float(y), 4)
    except Exception:
        return None


def _valid_path_points_count(path):
    if not isinstance(path, list):
        return 0
    return sum(1 for p in path if isinstance(p, (list, tuple)) and len(p) >= 2)


def reset_local_recorded_path():
    global last_path_point, last_path_append_time, latest_data
    latest_data["recorded_path"] = []
    latest_data["path_debug"] = {
        "mode": "reset",
        "points": 0,
        "last_xy": None,
        "note": "local recorded_path cleared"
    }
    last_path_point = None
    last_path_append_time = 0.0


def append_local_recorded_path(x, y, force=False):
    global last_path_point, last_path_append_time, latest_data
    xy = _sanitize_xy(x, y)
    if xy is None:
        return False

    x, y = xy
    now = time.time()
    path = latest_data.get("recorded_path")
    if not isinstance(path, list):
        path = []
        latest_data["recorded_path"] = path

    if not path:
        path.append([x, y])
        last_path_point = (x, y)
        last_path_append_time = now
        return True

    if last_path_point is None:
        for p in reversed(path):
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                last_path_point = (float(p[0]), float(p[1]))
                break
        if last_path_point is None:
            last_path_point = (x, y)

    dx = x - last_path_point[0]
    dy = y - last_path_point[1]
    dist = (dx * dx + dy * dy) ** 0.5

    if dist >= PATH_JUMP_BREAK_DIST:
        path.append(None)
        path.append([x, y])
        if len(path) > PATH_MAX_POINTS:
            del path[:-PATH_MAX_POINTS]
        last_path_point = (x, y)
        last_path_append_time = now
        return True

    if force or dist >= PATH_APPEND_MIN_DIST:
        path.append([x, y])
        if len(path) > PATH_MAX_POINTS:
            del path[:-PATH_MAX_POINTS]
        last_path_point = (x, y)
        last_path_append_time = now
        return True

    if now - last_path_append_time >= PATH_FORCE_APPEND_INTERVAL:
        path.append([x, y])
        if len(path) > PATH_MAX_POINTS:
            del path[:-PATH_MAX_POINTS]
        last_path_append_time = now
        return True
    return False


def maybe_merge_remote_recorded_path(remote_path):
    global last_path_point, latest_data
    if not isinstance(remote_path, list):
        return False

    local_path = latest_data.get("recorded_path", [])
    local_count = _valid_path_points_count(local_path)

    cleaned = []
    for p in remote_path:
        if p is None:
            cleaned.append(None)
            continue
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            xy = _sanitize_xy(p[0], p[1])
            if xy is not None:
                cleaned.append([xy[0], xy[1]])

    remote_count = _valid_path_points_count(cleaned)
    if remote_count < 3 or remote_count <= local_count + 2:
        return False

    if len(cleaned) > PATH_MAX_POINTS:
        cleaned = cleaned[-PATH_MAX_POINTS:]

    latest_data["recorded_path"] = cleaned
    for p in reversed(cleaned):
        if isinstance(p, list) and len(p) >= 2:
            last_path_point = (float(p[0]), float(p[1]))
            break
    return True


soil_history_queue = {
    "labels": [], "temp": [], "moisture": [], "nitrogen": [], "phosphorus": [], "potassium": []
}
last_soil_timestamp = None
device_connected = False
current_device_info = {'ip': RK_IP, 'port': RK_PORT}

DEFAULT_BASE_SAVE_PATH = r"C:\Users\26405\uploads"
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save_path_config.json")


def load_base_save_path():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            path = str(cfg.get("base_save_path", "")).strip()
            if path: return path
    except Exception as e:
        print(f"⚠️ [路径配置] 读取失败，使用默认路径: {e}")
    return DEFAULT_BASE_SAVE_PATH


def save_base_save_path(path):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"base_save_path": path}, f, ensure_ascii=False, indent=2)


BASE_SAVE_PATH = load_base_save_path()
PENDING_FOLDER = os.path.join(BASE_SAVE_PATH, "pending_reports")
MAPS_LOCAL_FOLDER = os.path.join(BASE_SAVE_PATH, "maps")

os.makedirs(BASE_SAVE_PATH, exist_ok=True)
os.makedirs(PENDING_FOLDER, exist_ok=True)
os.makedirs(MAPS_LOCAL_FOLDER, exist_ok=True)

app.config['PENDING_FOLDER'] = PENDING_FOLDER

AI_CONFIG = {
    'api_key': os.environ.get('DASHSCOPE_API_KEY', ''),
    'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
    'model': 'qwen3.6-35b-a3b'
}

ROUTE_FILE_NAME = "route.txt"
if not os.path.exists(ROUTE_FILE_NAME):
    try:
        with open(ROUTE_FILE_NAME, "w", encoding="utf-8") as f:
            f.write("# 自动化蛇形巡航基本配置参数\nROW_LENGTH=6.0\nROW_SPACING=0.70\nTOTAL_ROWS=4\nTURN_DIR=1\n")
    except: pass


@app.route('/')
def index(): return render_template('index.html')


@app.route('/api/save-path', methods=['GET'])
def get_save_path():
    return jsonify({'success': True, 'path': BASE_SAVE_PATH})


@app.route('/api/save-path', methods=['POST'])
def set_save_path():
    global BASE_SAVE_PATH, PENDING_FOLDER, MAPS_LOCAL_FOLDER
    try:
        data = request.get_json(force=True) or {}
        new_path = str(data.get("path", "")).strip().strip('"')
        if not new_path:
            return jsonify({'success': False, 'error': '路径不能为空'}), 400

        new_path = os.path.abspath(os.path.expanduser(new_path))
        os.makedirs(new_path, exist_ok=True)
        
        BASE_SAVE_PATH = new_path
        PENDING_FOLDER = os.path.join(BASE_SAVE_PATH, "pending_reports")
        MAPS_LOCAL_FOLDER = os.path.join(BASE_SAVE_PATH, "maps")
        os.makedirs(PENDING_FOLDER, exist_ok=True)
        os.makedirs(MAPS_LOCAL_FOLDER, exist_ok=True)
        app.config['PENDING_FOLDER'] = PENDING_FOLDER

        save_base_save_path(BASE_SAVE_PATH)
        return jsonify({
            'success': True, 'path': BASE_SAVE_PATH,
            'pending_folder': PENDING_FOLDER, 'maps_folder': MAPS_LOCAL_FOLDER
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def safe_join_base(rel_path):
    rel_path = str(rel_path or "").strip().replace("/", os.sep).replace("\\", os.sep)
    full_path = os.path.abspath(os.path.join(BASE_SAVE_PATH, rel_path))
    base_abs = os.path.abspath(BASE_SAVE_PATH)
    if os.path.commonpath([base_abs, full_path]) != base_abs: return None
    return full_path


# 🚀 核心重构：后端智能提取文件名中的具体时间部分，并映射为 -1 与 -2 展示
@app.route('/api/sugarcane/history-videos', methods=['GET'])
def get_history_videos():
    try:
        file_list = []
        if os.path.exists(BASE_SAVE_PATH):
            for root, dirs, files in os.walk(BASE_SAVE_PATH):
                for f in files:
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.svg', '.json', '.txt')) and not f.startswith('save_path_config'):
                        full_p = os.path.join(root, f)
                        rel_path = os.path.relpath(full_p, BASE_SAVE_PATH)
                        rel_url_path = rel_path.replace(os.sep, "/")
                        
                        parent_dir = os.path.basename(root)
                        # 如果在Task子目录下，优先使用子目录作为大组名，否则使用根目录
                        group_tag = parent_dir if parent_dir.startswith("Task_") else "本地历史固化归档"
                        
                        # 核心逻辑：从文件名中提取时间戳，例如从 Video_Raw_20260705_195147 提取 195147
                        time_match = re.search(r'(\d{8})_(\d{6})', f)
                        if time_match:
                            t_date = time_match.group(1)  # 20260705
                            t_time = time_match.group(2)  # 195147
                            time_display = f"{t_time[0:2]}:{t_time[2:4]}:{t_time[4:6]}" # 19:51:47
                        else:
                            time_display = "未知时间"

                        # 核心判定：划分 -1（检测/专家版）与 -2（原始/基础巡检版）
                        f_lower = f.lower()
                        display_name = f
                        if f_lower.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            if 'box' in f_lower:
                                display_name = f"🎬 {time_display}-1 (AI动态识别流)"
                            else:
                                display_name = f"🌿 {time_display}-2 (高精原始录像)"
                        elif f_lower.endswith('.txt'):
                            if 'expert' in f_lower:
                                display_name = f"📄 {time_display}-1 (AI专家分析报告)"
                            else:
                                display_name = f"📊 {time_display}-2 (巡检原始数据表)"

                        stat = os.stat(full_p)
                        file_list.append({
                            'filename': f,
                            'display_name': display_name,  # 注入前端选择框看到的干净文字
                            'rel_path': rel_url_path,
                            'group_tag': group_tag, 
                            'url': '/api/sugarcane/play-video?path=' + quote(rel_url_path, safe='/'),
                            'size': f"{stat.st_size / (1024 * 1024):.2f} MB" if not f_lower.endswith(('.json', '.txt')) else f"{stat.st_size / 1024:.1f} KB",
                            'time': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                            'is_json_map': f_lower.endswith('.json'),
                            'is_svg': f_lower.endswith('.svg')
                        })
        file_list.sort(key=lambda x: x['time'], reverse=True)
        return jsonify({'success': True, 'base_path': BASE_SAVE_PATH, 'videos': file_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sugarcane/play-video', methods=['GET'])
def play_video():
    try:
        rel_path = request.args.get("path", "")
        full_path = safe_join_base(rel_path)
        if full_path is None or not os.path.exists(full_path):
            return jsonify({'success': False, 'error': '文件未找到'}), 404

        ext = os.path.splitext(full_path)[1].lower()
        mime_map = {
            '.mp4': 'video/mp4', '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime', '.mkv': 'video/x-matroska',
            '.svg': 'image/svg+xml', '.json': 'application/json',
            '.txt': 'text/plain; charset=utf-8' 
        }
        return send_file(full_path, mimetype=mime_map.get(ext, 'application/octet-stream'), conditional=True)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sugarcane/open-local', methods=['POST'])
def open_local_app():
    try:
        data = request.get_json() or {}
        rel_path = data.get("path", "")
        full_path = safe_join_base(rel_path)
        
        if full_path is None or not os.path.exists(full_path):
            return jsonify({'success': False, 'error': '文件不存在'}), 404
            
        if platform.system() == "Windows":
            os.startfile(full_path)
        else:
            opener = "open" if platform.system() == "Darwin" else "xdg-open"
            subprocess.call([opener, full_path])
            
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sugarcane/soil-data', methods=['GET'])
def get_soil_data():
    global soil_history_queue
    if not soil_history_queue["labels"]:
        return jsonify({'success': True, 'data': {"labels": ["5-19", "5-20"], "temp": [25, 26], "moisture": [68, 65], "nitrogen": [142, 140], "phosphorus": [24, 25], "potassium": [180, 182]}})
    return jsonify({'success': True, 'data': soil_history_queue})


@app.route('/api/upload-report', methods=['POST'])
def upload_report():
    try:
        if 'file' not in request.files: return jsonify({'success': False, 'error': 'No file'}), 400
        file = request.files['file']
        if file.filename == '': return jsonify({'success': False, 'error': 'No filename'}), 400
        
        timestamp = ""
        parts = file.filename.replace(".txt", "").replace(".svg", "").replace(".json", "").split("_")
        for p in parts:
            if p.isdigit() and len(p) == 8:
                idx = parts.index(p)
                if idx + 1 < len(parts): timestamp = f"{p}_{parts[idx+1]}"
        
        if timestamp:
            task_folder = os.path.join(BASE_SAVE_PATH, f"Task_{timestamp}")
            os.makedirs(task_folder, exist_ok=True)
            save_path = os.path.join(task_folder, file.filename)
        else:
            save_path = os.path.join(BASE_SAVE_PATH, file.filename)
            
        file.save(save_path)
        return jsonify({'success': True, 'path': save_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def handle_assistant_chat():
    try:
        data = request.get_json(force=True) or {}
        msg = data.get("message", "")
        if not msg: return jsonify({"result": "输入不能为空"})
        prompt_messages = [
            {"role": "system", "content": "你是一位资深的精细化甘蔗农业和农机工程专家。请给用户提供精准、具备可操作性的现场作业建议与分析诊断。"},
            {"role": "user", "content": f"当前小车传感器摘要数据：当前位置坐标X={latest_data['car_world_x']}, Y={latest_data['car_world_y']}，电池电压={latest_data['battery_voltage']}V，最新土壤肥力数据：{json.dumps(latest_data['f1_soil_data'], ensure_ascii=False)}。用户现场问题：{msg}"}
        ]
        ans = call_qwen_api(prompt_messages)
        return jsonify({"result": ans})
    except Exception as e:
        return jsonify({"result": f"农技专家大脑神经网络异常: {str(e)}"})


def call_qwen_api(messages):
    try:
        if not AI_CONFIG.get("api_key"):
            return "智能助理异常: 未配置 DASHSCOPE_API_KEY 环境变量"
        response = requests.post(
            AI_CONFIG['api_url'], 
            headers={'Authorization': f'Bearer {AI_CONFIG["api_key"]}', 'Content-Type': 'application/json'},
            json={'model': AI_CONFIG['model'], 'messages': messages, 'max_tokens': 2048, 'temperature': 0.7}, 
            timeout=30
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e: 
        return f"大模型通信异常: {str(e)}"


@app.route("/api/gcs/latest")
def gcs_latest():
    global latest_data, RK_IP, device_connected
    status_block = latest_data.get("route_status", {})
    return jsonify({
        "rk_ip": RK_IP, "connected": device_connected,
        "car_world_x": float(latest_data.get("car_world_x", 0.0)), "car_world_y": float(latest_data.get("car_world_y", 0.0)), "car_yaw": float(latest_data.get("car_yaw", 0.0)),
        "obstacle_in_1m": bool(latest_data.get("obstacle_in_1m", False)), "recorded_path": latest_data.get("recorded_path", []), "path_debug": latest_data.get("path_debug", {}),
        "permanent_obstacles": latest_data.get("permanent_obstacles", []), "semantic_markers": latest_data.get("semantic_markers", []),
        "ai_running": bool(latest_data.get("ai_running", False)), "ai_counts": latest_data.get("ai_counts", [0] * 7),
        "speed": float(latest_data.get("speed", 0.20)), "turn_speed": float(latest_data.get("turn_speed", 1.2)),
        "min_dist": float(latest_data.get("min_dist", 0.15)), "max_dist": float(latest_data.get("max_dist", 8.0)),
        "nav_running": bool(latest_data.get("nav_running", False)), "detection_side": latest_data.get("detection_side", "right"),
        "show_lidar_points": bool(latest_data.get("show_lidar_points", True)), "battery_voltage": float(latest_data.get("battery_voltage", 12.0)),
        "spray_master_switch": bool(latest_data.get("spray_master_switch", False)), "route_status": status_block, "route_synced": status_block.get("route_synced", False)
    })


@app.route("/api/gcs/cmd/<cmd>", methods=["POST"])
def send_cmd_via_post(cmd):
    cmd_upper = cmd.strip().upper()
    if cmd_upper in ["CLEAR_MAP", "RESET_MAP", "CLEAR"]: clear_local_map_cache()
    if cmd_upper in ["STOP", "STOP_MOVE"]: threading.Thread(target=stop_burst_worker, args=(cmd,), daemon=True).start()
    else: threading.Thread(target=send_cmd_to_rk, args=(cmd,), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/gcs/set-params", methods=["POST"])
def set_gcs_params():
    global latest_data
    try:
        data = request.get_json()
        payload = {"type": "SET_PARAMS", "speed": data.get("speed", 0.20), "turn_speed": data.get("turn_speed", 1.2), "min_dist": data.get("min_dist", 0.15), "max_dist": data.get("max_dist", 8.0), "detection_side": data.get("detection_side", "right"), "show_lidar_points": data.get("show_lidar_points", True)}
        for key in ["speed", "turn_speed", "min_dist", "max_dist", "detection_side", "show_lidar_points"]:
            if key in data: latest_data[key] = data[key]
        threading.Thread(target=send_cmd_to_rk, args=(json.dumps(payload),), daemon=True).start()
        return jsonify({"success": True})
    except Exception as e: return jsonify({"success": False, "error": str(e)})


@app.route("/api/gcs/update-row-length", methods=["POST"])
def update_row_len():
    val = request.get_json().get("value", 6.0)
    threading.Thread(target=send_cmd_to_rk, args=(f"SET_ROW_LEN {val}",), daemon=True).start()
    return jsonify({"success": True})

@app.route("/api/gcs/update-row-spacing", methods=["POST"])
def update_row_space():
    val = request.get_json().get("value", 0.70)
    threading.Thread(target=send_cmd_to_rk, args=(f"SET_ROW_SPACE {val}",), daemon=True).start()
    return jsonify({"success": True})

@app.route("/api/gcs/update-total-rows", methods=["POST"])
def update_total_rows():
    val = request.get_json().get("value", 4)
    threading.Thread(target=send_cmd_to_rk, args=(f"SET_TOTAL_ROWS {val}",), daemon=True).start()
    return jsonify({"success": True})

@app.route("/api/gcs/update-turn-dir", methods=["POST"])
def update_turn_dir():
    val = request.get_json().get("value", 1)
    threading.Thread(target=send_cmd_to_rk, args=(f"SET_TURN_DIR {val}",), daemon=True).start()
    return jsonify({"success": True})

@app.route("/api/gcs/start-patrol", methods=["POST"])
def start_patrol_params():
    threading.Thread(target=send_cmd_to_rk, args=("START_PATROL",), daemon=True).start()
    return jsonify({"success": True})

@app.route("/api/gcs/stop-patrol", methods=["POST"])
def stop_patrol_params():
    threading.Thread(target=send_cmd_to_rk, args=("STOP_PATROL",), daemon=True).start()
    return jsonify({"success": True})


def auto_discovery_worker():
    global RK_IP, current_device_info, device_connected
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', UDP_PORT))
    while True:
        try:
            data, _ = s.recvfrom(1024)
            msg = data.decode('utf-8', errors="ignore")
            if "I_AM_RK3588_IP:" in msg:
                new_ip = msg.split(":")[1].strip()
                if new_ip != RK_IP or not device_connected:
                    RK_IP = new_ip
                    current_device_info = {'ip': RK_IP, 'port': RK_PORT}
                    socketio.emit('current_device', current_device_info)
            time.sleep(1)
        except: time.sleep(5)


def send_cmd_to_rk(cmd_str):
    if not cmd_str.endswith('\n') and not cmd_str.startswith('{'): cmd_str += '\n'
    cmd_socket = None
    try:
        cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cmd_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        cmd_socket.settimeout(0.2)
        cmd_socket.connect((RK_IP, RK_CMD_PORT))
        cmd_socket.sendall(cmd_str.encode('utf-8'))
        return True
    except: return False
    finally:
        if cmd_socket:
            try: cmd_socket.close()
            except: pass


def stop_burst_worker(cmd):
    for i in range(2):
        success = send_cmd_to_rk(cmd)
        if success and i == 0: time.sleep(0.008)


def clear_local_map_cache():
    global latest_data, map_clear_guard_until
    map_clear_guard_until = time.time() + 1.5
    latest_data["permanent_obstacles"] = []
    latest_data["semantic_markers"] = []
    reset_local_recorded_path()
    latest_data["ai_counts"] = [0] * 7
    socketio.emit('telemetry_update', latest_data)


def master_compound_stream_worker():
    global device_connected, latest_data, soil_history_queue, last_soil_timestamp
    while True:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(3.0)
        try:
            client_socket.connect((RK_IP, RK_PORT))
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            device_connected = True
            socketio.emit('device_connected', {'ip': RK_IP, 'port': RK_PORT})
            client_socket.settimeout(None)
            
            while True:
                img_len_bytes = client_socket.recv(4)
                if not img_len_bytes or len(img_len_bytes) < 4: break
                img_size = int.from_bytes(img_len_bytes, 'big')
                
                img_data = b''
                while len(img_data) < img_size:
                    chunk = client_socket.recv(img_size - len(img_data))
                    if not chunk: break
                    img_data += chunk

                txt_len_bytes = client_socket.recv(4)
                if not txt_len_bytes or len(txt_len_bytes) < 4: break
                txt_size = int.from_bytes(txt_len_bytes, 'big')
                
                txt_data = b''
                while len(txt_data) < txt_size:
                    chunk = client_socket.recv(txt_size - len(txt_data))
                    if not chunk: break
                    txt_data += chunk

                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                    socketio.emit('video_frame', {'image': base64.b64encode(buf).decode('utf-8')})

                try:
                    pack = json.loads(txt_data.decode('utf-8'))
                    if isinstance(pack, dict) and "car_world_x" in pack:
                        latest_data["car_world_x"] = pack.get("car_world_x", 0.0)
                        latest_data["car_world_y"] = pack.get("car_world_y", 0.0)
                        latest_data["car_yaw"] = pack.get("car_yaw", 0.0)
                        latest_data["obstacle_in_1m"] = pack.get("obstacle_in_1m", False)
                        
                        if time.time() < map_clear_guard_until:
                            reset_local_recorded_path()
                            latest_data["permanent_obstacles"] = []
                            latest_data["semantic_markers"] = []
                            latest_data["ai_counts"] = [0] * 7
                        else:
                            remote_path = pack.get("recorded_path", None)
                            accepted_remote = maybe_merge_remote_recorded_path(remote_path)
                            if not accepted_remote:
                                append_local_recorded_path(latest_data["car_world_x"], latest_data["car_world_y"])

                            latest_data["permanent_obstacles"] = pack.get("permanent_obstacles", [])
                            latest_data["semantic_markers"] = pack.get("semantic_markers", [])
                            latest_data["ai_counts"] = pack.get("ai_counts", [0] * 7)
                            
                        latest_data["ai_running"] = pack.get("ai_running", False)
                        latest_data["battery_voltage"] = pack.get("battery_voltage", 12.0)
                        latest_data["spray_master_switch"] = pack.get("spray_master_switch", False)
                        latest_data["route_status"] = pack.get("route_status", {})
                        latest_data["nav_running"] = pack.get("nav_running", False)

                        soil_pack = pack.get("f1_soil_data")
                        if soil_pack and isinstance(soil_pack, dict):
                            latest_data["f1_soil_data"] = soil_pack
                            current_stamp = f"{soil_pack.get('nitrogen', 0)}_{soil_pack.get('moisture', 0)}"
                            if current_stamp != last_soil_timestamp:
                                last_soil_timestamp = current_stamp
                                soil_history_queue["labels"].append(datetime.now().strftime('%H:%M:%S'))
                                soil_history_queue["temp"].append(float(soil_pack.get("temp", 0)))
                                soil_history_queue["moisture"].append(float(soil_pack.get("moisture", 0)))
                                soil_history_queue["nitrogen"].append(float(soil_pack.get("nitrogen", 0)))
                                soil_history_queue["phosphorus"].append(float(soil_pack.get("phosphorus", 0)))
                                soil_history_queue["potassium"].append(float(soil_pack.get("potassium", 0)))
                                if len(soil_history_queue["labels"]) > 6:
                                    for k in soil_history_queue: soil_history_queue[k].pop(0)

                        socketio.emit('telemetry_update', latest_data)
                except: pass
                socketio.sleep(0.001)
        except:
            if device_connected: device_connected = False; socketio.emit('device_disconnected')
            socketio.sleep(0.5)
        finally:
            client_socket.close()


def background_assets_receiver():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(('0.0.0.0', ASSETS_RECV_PORT))
        server.listen(10)
        print(f"📥 [数据管理中心] 资产落盘通道就绪，监听端口: {ASSETS_RECV_PORT}")
    except Exception as e:
        print(f"❌ 通道绑定失败: {e}")
        return

    while True:
        try:
            conn, _ = server.accept()
            header = b""
            while b"\n" not in header:
                chunk = conn.recv(1)
                if not chunk: break
                header += chunk
            if not header:
                conn.close()
                continue
                
            file_name, file_size = header.decode('utf-8').strip().split(":")
            file_size = int(file_size)
            
            timestamp = ""
            parts = file_name.replace(".txt", "").replace(".mp4", "").replace(".svg", "").replace(".json", "").split("_")
            for p in parts:
                if p.isdigit() and len(p) == 8:
                    idx = parts.index(p)
                    if idx + 1 < len(parts): timestamp = f"{p}_{parts[idx+1]}"
            
            if timestamp:
                task_folder = os.path.join(BASE_SAVE_PATH, f"Task_{timestamp}")
                os.makedirs(task_folder, exist_ok=True)
                out_path = os.path.join(task_folder, file_name)
            else:
                out_path = os.path.join(BASE_SAVE_PATH, file_name)
                
            received = 0
            with open(out_path, 'wb') as f:
                while received < file_size:
                    chunk = conn.recv(min(65536, file_size - received))
                    if not chunk: break
                    f.write(chunk)
                    received += len(chunk)
            conn.close()
        except: pass


def auto_open_browser(): eventlet.sleep(1.5); webbrowser.open("http://127.0.0.1:5000")

if __name__ == '__main__':
    threading.Thread(target=auto_discovery_worker, daemon=True).start()
    threading.Thread(target=background_assets_receiver, daemon=True).start()
    socketio.start_background_task(master_compound_stream_worker)
    socketio.start_background_task(auto_open_browser)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)