#!/usr/bin/env python3
import cv2
import socket
import time
import serial
import struct
import os
import sys
import threading
import json
import math
import queue
import subprocess
import numpy as np
from datetime import datetime

# 🤫 静音 rknn 刷屏
os.environ["RKNN_LOG_LEVEL"] = "E"

from rknnpool.rknnpool_ld import rknnPoolExecutor
from func.func_yolov8_optimize import myFunc

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32   
from std_msgs.msg import String  
from nav_msgs.msg import Odometry  

UDP_PORT = 9999
serial_lock = threading.Lock()
obstacles_lock = threading.Lock() 
pose_lock = threading.Lock()       
semantic_lock = threading.Lock()   

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except: return "127.0.0.1"

def broadcast_worker():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    print("📡 RK3588 身份广播服务已启动...")
    while True:
        try:
            local_ip = get_local_ip()
            s.sendto(f"I_AM_RK3588_IP:{local_ip}".encode('utf-8'), ('<broadcast>', UDP_PORT))
            time.sleep(3)
        except: time.sleep(3)

threading.Thread(target=broadcast_worker, daemon=True).start()


class RosGameReceiver(Node):
    def __init__(self, gcs_service):
        super().__init__('gcs_game_receiver')
        self.gcs = gcs_service
        self.scan_sub = self.create_subscription(LaserScan, '/scan_filtered', self.lidar_callback, 10)
        self.vcc_sub = self.create_subscription(Float32, '/battery_voltage', self.vcc_callback, 10)
        self.odom_direct_sub = self.create_subscription(Odometry, '/odom_raw', self.odom_direct_callback, 10)
        self.stm_sub = self.create_subscription(String, '/stm_data', self.stm_data_callback, 10)

    def lidar_callback(self, msg):
        self.gcs.real_lidar_data = list(msg.ranges)[::-1]
        self.gcs.angle_min = msg.angle_min
        self.gcs.angle_increment = msg.angle_increment

    def vcc_callback(self, msg):
        self.gcs.real_battery_vcc = msg.data

    def odom_direct_callback(self, msg):
        real_x = -msg.pose.pose.position.x
        real_y = -msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        real_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        with pose_lock:
            self.gcs.smooth_x = real_x
            self.gcs.smooth_y = real_y
            fast_alpha_yaw = 0.85
            smooth_sin = math.sin(self.gcs.smooth_yaw) + fast_alpha_yaw * (math.sin(real_yaw) - math.sin(self.gcs.smooth_yaw))
            smooth_cos = math.cos(self.gcs.smooth_yaw) + fast_alpha_yaw * (math.cos(real_yaw) - math.cos(self.gcs.smooth_yaw))
            self.gcs.smooth_yaw = math.atan2(smooth_sin, smooth_cos)
            self.gcs.car_world_x = self.gcs.smooth_x
            self.gcs.car_world_y = self.gcs.smooth_y
            self.gcs.car_yaw = self.gcs.smooth_yaw

    def stm_data_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            data_type = payload.get("data_type", "")
            if data_type != "soil_fertility": return
            self.gcs.latest_soil_packet = {
                "temp": round(float(payload.get("temp", 0.0)), 1),
                "moisture": round(float(payload.get("moisture", 0.0)), 1),
                "nitrogen": int(payload.get("n", 0)),
                "phosphorus": int(payload.get("p", 0)),
                "potassium": int(payload.get("k", 0)),
                "timestamp": int(time.time()),
                "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "ros_stm_data_parsed_json"
            }
        except: pass


class AgriRobotHeadlessMaster:
    def __init__(self):
        self.BASE_DIR = "/home/elf/Desktop/Agriculture_Records"
        self.DEFAULT_ROUTE_PATH = "/home/elf/Desktop/Agriculture_Routes/route.txt"
        os.makedirs(self.BASE_DIR, exist_ok=True)

        self.CMD_FORWARD      = bytes.fromhex("AA 02 01")
        self.CMD_BACKWARD     = bytes.fromhex("AA 03 01")
        self.CMD_STOP         = bytes.fromhex("AA 07 01")
        self.CMD_ROTATE_LEFT  = bytes.fromhex("AA 09 01")
        self.CMD_ROTATE_RIGHT = bytes.fromhex("AA 08 01")
        self.CMD_CAM_LEFT     = bytes.fromhex("AA 2A 01") 
        self.CMD_CAM_RIGHT    = bytes.fromhex("AA 2B 01") 
        self.CMD_SOIL_LEFT    = bytes.fromhex("AA 37 01") 
        self.CMD_SOIL_RIGHT   = bytes.fromhex("AA 38 01") 
        
        try:
            self.ser = serial.Serial('/dev/ttyS9', 115200, timeout=0)
            print("✅ 串口已连接")
        except: 
            self.ser = None
            print("❌ 串口未连接")

        self.car_world_x, self.car_world_y, self.car_yaw = 0.0, 0.0, 0.0
        self.smooth_x, self.smooth_y, self.smooth_yaw = 0.0, 0.0, 0.0
        self.real_battery_vcc = 12.0                                                 
        self.real_lidar_data = []
        self.angle_min, self.angle_increment = 0.0, 0.0
        self.obstacle_in_1m = False 
        self.recorded_path = []      
        self.semantic_markers = []      
        self.sprayed_weed_points = []
        self.SPRAYED_WEED_DIST = 0.30
        self.map_clear_seq = 0
        self.map_clear_force_empty_until = 0.0

        # 🗺️ 运动实时坐标高精缓存队列
        self.path_log = []

        self.current_camera_dir = 0     # 0:正前, 1:左边, 2:右边
        self.spray_master_switch = True 
        self.is_spraying_blocking = False
        self.spray_block_timeout = 0.0
        self.weed_trigger_cooldown = 0.0
        self.WEED_COOLDOWN_TIME = 2.0       

        self.show_lidar_points = True   
        self.target_zone_min = 200      
        self.target_zone_max = 440      
        self.latest_yolo_cx = -1
        self.latest_yolo_cls = -1
        self.lidar_min_dist = 0.15
        self.lidar_max_dist = 8.0
        self.PC_IP = "192.168.0.168"

        self.task_running = False  
        self.timestamp = ""
        self.SAVE_DIR = ""
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.raw_video_writer = None
        self.box_video_writer = None
        
        self.CLASSES_NAMES = ("甘蔗（幼苗）", "杂草（薇甘菊）", "杂草（胜红蓟）", "杂草（掌叶鱼黄草）", "杂草（三叶草）", "杂草（绿辣子）","杂草（篱栏网）", "甘蔗（半熟）", "甘蔗（全熟）", "褐条病", "霜霉病", "锈病")
        self.stats = {"counts": [0]*12, "details": []} 
        self.compound_asset_queue = queue.Queue(maxsize=1) 
        self.clients = []

        self.is_moving_forward = False
        self.is_moving_backward = False  
        self.lock_target_yaw = 0.0  
        self.kp = 1.6               
        self.kd = 0.5               
        self.last_yaw_error = 0.0   
        self.pid_loop_counter = 0   
        self.macro_queue = []       
        self.current_action = "STOP"
        self.action_start_x, self.action_start_y, self.action_start_yaw = 0.0, 0.0, 0.0

        # 🛠️ 新增控制核心变量：启动热身计数器与巡航打断恢复标记
        self.start_ignore_counter = 0
        self.was_auto_driving_before_spray = False

        self.latest_soil_packet = None
        self.route_executor = None

        threading.Thread(target=self.compound_socket_sender_worker, daemon=True).start()
        threading.Thread(target=self.accept_worker, daemon=True).start()
        threading.Thread(target=self.cmd_listener_worker, daemon=True).start()
        threading.Thread(target=self.game_loop_worker, daemon=True).start() 

    def clear_map_layers(self, reason="REMOTE_CLEAR"):
        with semantic_lock:
            self.semantic_markers.clear()
            self.sprayed_weed_points.clear()
            self.stats["counts"] = [0] * 12
            self.stats["details"].clear()
        with obstacles_lock:
            self.real_lidar_data.clear()

        self.path_log.clear()
        self.latest_yolo_cx = -1
        self.latest_yolo_cls = -1
        self.map_clear_seq += 1
        self.map_clear_force_empty_until = time.time() + 2.0
        print(f"🧹 [指令总线] 本地地图/植物标记已清空，seq={self.map_clear_seq}, reason={reason}")

    def has_sprayed_weed_near(self, wx, wy, threshold=None):
        if threshold is None:
            threshold = self.SPRAYED_WEED_DIST
        with semantic_lock:
            sprayed_snapshot = list(self.sprayed_weed_points)
        for sx, sy in sprayed_snapshot:
            if math.hypot(wx - sx, wy - sy) < threshold:
                return True
        return False

    def mark_weed_sprayed(self, wx, wy):
        with semantic_lock:
            self.sprayed_weed_points.append([wx, wy])
            if len(self.sprayed_weed_points) > 300:
                self.sprayed_weed_points = self.sprayed_weed_points[-300:]

    def safe_serial_send(self, cmd_bytes):
        if self.ser and self.ser.is_open:
            with serial_lock: 
                try: 
                    self.ser.write(cmd_bytes)
                    self.ser.flush()
                    time.sleep(0.005) 
                except: pass

    def send_spray_command(self, direction: int, duration_s: int):
        direction = max(0, min(2, direction))
        dist_bcd = ((25 // 10) << 4) | (25 % 10)
        duration_s = max(1, min(5, duration_s))
        cmd_bytes = bytes([0xAA, direction, dist_bcd, duration_s, 0x01])
        self.safe_serial_send(cmd_bytes)

    def change_macro_action(self, next_action):
        if not self.task_running: self.current_action = next_action
        if self.current_action == "FORWARD":
            dist = math.hypot(self.car_world_x - self.action_start_x, self.car_world_y - self.action_start_y)
            if dist > 0.04: self.macro_queue.append({"action": "FORWARD", "value": dist})
        elif self.current_action in ["LEFT", "RIGHT"]:
            diff = math.atan2(math.sin(self.car_yaw - self.action_start_yaw), math.cos(self.car_yaw - self.action_start_yaw))
            if abs(diff) > 0.03: self.macro_queue.append({"action": "TURN", "value": diff})
        self.action_start_x, self.action_start_y, self.action_start_yaw = self.car_world_x, self.car_world_y, self.car_yaw
        self.current_action = next_action

    def handle_web_cmd_string(self, cmd_str):
        cmd_str = cmd_str.strip()
        if not cmd_str: return
        
        if cmd_str == "CAM_LEFT": self.current_camera_dir = 1; self.safe_serial_send(self.CMD_CAM_LEFT); return
        if cmd_str == "CAM_RIGHT": self.current_camera_dir = 2; self.safe_serial_send(self.CMD_CAM_RIGHT); return
        if cmd_str == "RESET_ARM":
            self.current_camera_dir = 0
            self.safe_serial_send(bytes.fromhex("AA 28 01"))
            time.sleep(0.04); self.safe_serial_send(bytes.fromhex("AA 26 01"))
            time.sleep(0.04); self.safe_serial_send(bytes.fromhex("AA 2C 01"))  
            return
        if cmd_str == "SOIL_LEFT": self.safe_serial_send(self.CMD_SOIL_LEFT); return
        if cmd_str == "SOIL_RIGHT": self.safe_serial_send(self.CMD_SOIL_RIGHT); return
        if cmd_str == "SPRAY_ENABLE_TRUE": self.spray_master_switch = True; return
        if cmd_str == "SPRAY_ENABLE_FALSE": self.spray_master_switch = False; return

        clear_cmd = cmd_str.strip().upper().replace("-", "_")
        if clear_cmd in ["CLEAR_MAP", "RESET_MAP", "CLEAR", "MAP_CLEAR", "CLEAR_MARKERS", "CLEAR_MARKER", "CLEAR_SEMANTIC", "RESET_CLEAR"]:
            self.clear_map_layers(reason=clear_cmd)
            return

        if cmd_str.startswith("SET_ROW_LEN ") and self.route_executor:
            try: self.route_executor.row_length = float(cmd_str.split(" ")[1])
            except: pass
            return
        if cmd_str.startswith("SET_ROW_SPACE ") and self.route_executor:
            try: self.route_executor.row_spacing = float(cmd_str.split(" ")[1])
            except: pass
            return
        if cmd_str.startswith("SET_TOTAL_ROWS ") and self.route_executor:
            try: self.route_executor.total_rows = int(cmd_str.split(" ")[1])
            except: pass
            return
        if cmd_str.startswith("SET_TURN_DIR ") and self.route_executor:
            try: self.route_executor.turn_dir = int(cmd_str.split(" ")[1])
            except: pass
            return

        if cmd_str == "START_PATROL" and self.route_executor:
            # 🏁 启动自动巡航，赋予 10 帧（0.5秒）的保护期压制雷达和D项抖动
            self.start_ignore_counter = 10
            self.was_auto_driving_before_spray = False
            self.route_executor.start(
                row_length=getattr(self.route_executor, 'row_length', 6.0),
                row_spacing=getattr(self.route_executor, 'row_spacing', 0.70),
                total_rows=getattr(self.route_executor, 'total_rows', 4),
                turn_dir=getattr(self.route_executor, 'turn_dir', 1)
            )
            return
        if cmd_str == "STOP_PATROL" and self.route_executor:
            self.was_auto_driving_before_spray = False
            self.route_executor.stop(finish_report=False, reason="REMOTE_WEB_STOP")
            return

        if cmd_str.startswith("{"):
            try:
                js = json.loads(cmd_str)
                if str(js.get("type", "")).upper() == "SET_PARAMS":
                    self.lidar_min_dist = float(js.get("min_dist", self.lidar_min_dist))
                    self.lidar_max_dist = float(js.get("max_dist", self.lidar_max_dist))
                    self.show_lidar_points = bool(js.get("show_lidar_points", self.show_lidar_points))
                return
            except: return

        cmd_upper = cmd_str.upper()
        if cmd_upper in ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"] and self.route_executor and self.route_executor.is_running():
            self.route_executor.stop(finish_report=False, reason="MANUAL_OVERRIDE")

        if cmd_upper == "FORWARD":
            self.change_macro_action("FORWARD")
            if not self.is_moving_forward:
                self.lock_target_yaw = self.car_yaw
                self.last_yaw_error = 0.0
                self.is_moving_forward = True
                self.is_moving_backward = False
                self.start_ignore_counter = 10  # 🚀 手动点按前进也给予 10 帧防抖动热身期
            self.safe_serial_send(self.CMD_FORWARD)
        elif cmd_upper == "BACKWARD":
            self.change_macro_action("STOP")
            if not self.is_moving_backward:
                self.lock_target_yaw = self.car_yaw
                self.last_yaw_error = 0.0
                self.is_moving_backward = True
                self.is_moving_forward = False
                self.start_ignore_counter = 10  # 🚀 后退同样防抖
            self.safe_serial_send(self.CMD_BACKWARD)
        elif cmd_upper in ["LEFT_TURN", "LEFT"]:
            self.change_macro_action("LEFT"); self.is_moving_forward = False; self.is_moving_backward = False; self.safe_serial_send(self.CMD_ROTATE_LEFT)
        elif cmd_upper in ["RIGHT_TURN", "RIGHT"]:
            self.change_macro_action("RIGHT"); self.is_moving_forward = False; self.is_moving_backward = False; self.safe_serial_send(self.CMD_ROTATE_RIGHT)
        elif cmd_upper in ["STOP_MOVE", "STOP"]:
            self.change_macro_action("STOP"); self.is_moving_forward = False; self.is_moving_backward = False; self.safe_serial_send(self.CMD_STOP)
        elif cmd_upper == "START": self.start_task()
        elif cmd_upper == "STOP_TASK": self.stop_task()

    def cmd_listener_worker(self):
        cmd_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cmd_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        cmd_server.bind(('0.0.0.0', 8889)); cmd_server.listen(5)
        while True:
            try:
                conn, _ = cmd_server.accept()
                data = conn.recv(32768).decode('utf-8')
                if data.strip(): self.handle_web_cmd_string(data)
                conn.close()
            except: pass

    def accept_worker(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', 8888)); server.listen(5)
        while True:
            try: 
                conn, addr = server.accept()
                self.PC_IP = addr[0]
                conn.settimeout(0.3)
                self.clients.append(conn)
            except: pass

    def compound_socket_sender_worker(self):
        while True:
            try:
                asset = self.compound_asset_queue.get()
                if asset is None or not self.clients: continue
                frame, telemetry_pack = asset
                
                if self.route_executor:
                    telemetry_pack["route_status"] = self.route_executor.snapshot()
                    telemetry_pack["nav_running"] = self.route_executor.is_running()

                _, jpg = cv2.imencode('.webp', frame, [int(cv2.IMWRITE_WEBP_QUALITY), 40])
                jpg_bytes = jpg.tobytes()
                txt_bytes = json.dumps(telemetry_pack).encode('utf-8')
                
                compound_msg = len(jpg_bytes).to_bytes(4, 'big') + jpg_bytes + len(txt_bytes).to_bytes(4, 'big') + txt_bytes
                
                for conn in self.clients[:]:
                    try: 
                        conn.sendall(compound_msg)
                    except:
                        if conn in self.clients: self.clients.remove(conn)
                        try: conn.close()
                        except: pass
            except: time.sleep(0.01)

    def select_center_target_from_detections(self, detections):
        best_cx, best_cls = -1, -1
        min_dist_to_center = float('inf')
        for det in detections:
            try:
                if isinstance(det, dict):
                    box = det.get("box", [0,0,0,0])
                    cls_id = int(det.get("class", -1))
                else:
                    box = det[:4]
                    cls_id = int(det[5])
                cx = int((box[0] + box[2]) / 2)
                dist = abs(cx - 320)
                if dist < min_dist_to_center:
                    min_dist_to_center = dist
                    best_cx = cx
                    best_cls = cls_id
            except: pass
        return best_cx, best_cls

    def game_loop_worker(self):
        last_log_time = 0.0
        while True:
            try:
                # 💦 除草阻塞处理逻辑
                if self.is_spraying_blocking:
                    if time.time() < self.spray_block_timeout:
                        self.safe_serial_send(self.CMD_STOP); time.sleep(0.05); continue
                    else:
                        self.is_spraying_blocking = False
                        # 🔄 【核心业务新增】：喷药结束，如果发现之前是被打断的自动巡航，执行平滑状态重回
                        if self.was_auto_driving_before_spray:
                            print("🔄 [除草恢复] 喷药结束，恢复自动驾驶巡航状态。")
                            self.start_ignore_counter = 10  # 恢复行驶时，也重新给予10帧防抖保护
                            self.lock_target_yaw = self.car_yaw  # 重新校准锁定当前车身偏航角
                            self.last_yaw_error = 0.0
                            if self.route_executor:
                                # 触发底层状态机驱动履带/马达重新恢复运转
                                base_cmd = self.CMD_FORWARD
                                self.safe_serial_send(base_cmd)
                            self.was_auto_driving_before_spray = False

                if self.weed_trigger_cooldown > 0: self.weed_trigger_cooldown -= 0.05
                
                is_auto_driving = True if (self.route_executor and self.route_executor.is_running()) else False
                is_active_moving = (self.is_moving_forward or self.is_moving_backward or is_auto_driving)
                
                # 🗺️ 实时轨迹发生器
                if self.task_running and is_active_moving:
                    current_now = time.time()
                    if current_now - last_log_time >= 0.2:
                        self.path_log.append([round(self.car_world_x, 3), round(self.car_world_y, 3)])
                        last_log_time = current_now

                if is_active_moving and not self.is_spraying_blocking:
                    self.pid_loop_counter = (self.pid_loop_counter + 1) % 5
                    min_left_dist, min_right_dist = float('inf'), float('inf')
                    if self.real_lidar_data:
                        for i, distance in enumerate(self.real_lidar_data):
                            if 0.2 < distance < 1.8:
                                angle = self.angle_min + (i * self.angle_increment)
                                lx, ly = distance * math.cos(angle), distance * math.sin(angle) * (-1)
                                if 0.0 <= lx <= 0.6:
                                    if ly > 0 and distance < min_right_dist: min_right_dist = distance
                                    elif ly < 0 and distance < min_left_dist: min_left_dist = distance
                    
                    if min_left_dist != float('inf') and min_right_dist != float('inf'):
                        dist_error = min_left_dist - min_right_dist
                        
                        # 🛡️ 【消除抖动改进 1】：只有在热身保护期结束后，雷达才允许介入微调 lock_target_yaw
                        if self.start_ignore_counter <= 0:
                            if self.is_moving_backward: self.lock_target_yaw -= dist_error * 0.04
                            else: self.lock_target_yaw += dist_error * 0.04
                        else:
                            self.start_ignore_counter -= 1  # 保护期计数递减

                    yaw_error = math.atan2(math.sin(self.lock_target_yaw - self.car_yaw), math.cos(self.lock_target_yaw - self.car_yaw))
                    
                    # 🛡️ 【消除抖动改进 2】：在热身保护期内，强制砍掉微分 D 项，防止启动时突变产生冲击
                    if self.start_ignore_counter > 0:
                        pid_out = self.kp * yaw_error
                    else:
                        pid_out = self.kp * yaw_error + self.kd * ((yaw_error - self.last_yaw_error) / 0.05)
                        
                    self.last_yaw_error = yaw_error
                    error_threshold = max(0.0, min(50.0, abs(pid_out) * 10.0)) 
                    
                    base_cmd = self.CMD_FORWARD
                    if self.is_moving_backward: base_cmd = self.CMD_BACKWARD
                    
                    if not is_auto_driving:
                        if abs(yaw_error) < 0.025: self.safe_serial_send(base_cmd)
                        else:
                            should_rotate_left = (pid_out < 0) if self.is_moving_backward else (pid_out > 0)
                            if should_rotate_left: 
                                self.safe_serial_send(self.CMD_ROTATE_LEFT if self.pid_loop_counter < error_threshold else base_cmd)
                            else: 
                                self.safe_serial_send(self.CMD_ROTATE_RIGHT if self.pid_loop_counter < error_threshold else base_cmd)
                    else:
                        if abs(yaw_error) >= 0.025:
                            if pid_out > 0: self.safe_serial_send(self.CMD_ROTATE_LEFT)
                            else: self.safe_serial_send(self.CMD_ROTATE_RIGHT)

                cx, cls_id = self.latest_yolo_cx, self.latest_yolo_cls
                if 200 <= cx <= 440 and cls_id != -1:
                    label_name = self.CLASSES_NAMES[cls_id] if cls_id < len(self.CLASSES_NAMES) else "Unknown"
                    is_weed = "杂草" in label_name
                    
                    if self.real_lidar_data:
                        valid_wxs, valid_wys = [], []
                        min_single_d = float('inf')
                        best_single_wx, best_single_wy = None, None
                        best_single_ly = 0.0
                        
                        for i, distance in enumerate(self.real_lidar_data):
                            if self.lidar_min_dist < distance < 0.80: 
                                angle = self.angle_min + (i * self.angle_increment)
                                lx, ly = distance * math.cos(angle), distance * math.sin(angle) * (-1)
                                
                                matched = False
                                if self.current_camera_dir == 1:
                                    if ly < 0 and (abs(lx) / abs(ly)) < 0.26: matched = True
                                elif self.current_camera_dir == 2:
                                    if ly > 0 and (abs(lx) / abs(ly)) < 0.26: matched = True
                                else:                                  
                                    if ly > 0 and (abs(lx) / abs(ly)) < 0.26: matched = True
                                    elif ly < 0 and (abs(lx) / abs(ly)) < 0.26: matched = True
                                    
                                if matched and abs(lx) < 0.35:
                                    wx = self.car_world_x + (lx * math.cos(self.car_yaw) - ly * math.sin(self.car_yaw))
                                    wy = self.car_world_y + (lx * math.sin(self.car_yaw) + ly * math.cos(self.car_yaw))
                                    valid_wxs.append(wx); valid_wys.append(wy)
                                    if distance < min_single_d:
                                        min_single_d = distance
                                        best_single_wx = wx; best_single_wy = wy; best_single_ly = ly
                                        
                        best_wx, best_wy = None, None
                        if best_single_wx is not None:
                            best_wx = best_single_wx; best_wy = best_single_wy
                                
                        if best_wx is not None and best_wy is not None:
                            already_sprayed = self.has_sprayed_weed_near(best_wx, best_wy)
                            if is_weed and self.spray_master_switch and self.weed_trigger_cooldown <= 0 and not already_sprayed:
                                # 🔄 【核心业务修改】：如果除草被触发，且此时处于自动巡航状态，记录打断标记
                                if is_auto_driving:
                                    self.was_auto_driving_before_spray = True
                                    print("⏸️ [除草打断] 自动驾驶巡航被除草任务打断，已记录断点位置。")
                                else:
                                    self.was_auto_driving_before_spray = False

                                self.is_moving_forward = False
                                self.is_moving_backward = False
                                self.is_spraying_blocking = True
                                self.spray_block_timeout = time.time() + 10.0

                                self.safe_serial_send(self.CMD_STOP)
                                time.sleep(0.02)
                                cam_dir_code = self.current_camera_dir if self.current_camera_dir in [1, 2] else (2 if best_single_ly > 0 else 1)
                                self.send_spray_command(direction=cam_dir_code, duration_s=2)
                                self.mark_weed_sprayed(best_wx, best_wy)
                                self.weed_trigger_cooldown = self.WEED_COOLDOWN_TIME
                                print(f"💦 [除草触发] label={label_name}, once_point=({best_wx:.2f},{best_wy:.2f})")
                            elif is_weed and already_sprayed:
                                print(f"⏭️ [除草跳过] 杂草点附近已喷过: ({best_wx:.2f},{best_wy:.2f})")

                            is_too_close = False
                            with semantic_lock:
                                marker_snapshot = list(self.semantic_markers)
                            for marker in marker_snapshot:
                                if math.hypot(best_wx - marker[0], best_wy - marker[1]) < 0.30:
                                    is_too_close = True; break

                            if not is_too_close:
                                detected_ts = int(time.time())
                                detected_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(detected_ts))
                                side_str = "车身左侧" if self.current_camera_dir == 1 else ("车身右侧" if self.current_camera_dir == 2 else ("车身右侧" if best_single_ly > 0 else "车身左侧"))

                                with semantic_lock:
                                    self.semantic_markers.append([best_wx, best_wy, cls_id, label_name, detected_ts, detected_time, side_str])
                                    self.stats["details"].append({
                                        "time": detected_time, "label": label_name,
                                        "x": round(best_wx, 3), "y": round(best_wy, 3), "side": side_str
                                    })
                                    if cls_id < len(self.stats["counts"]): self.stats["counts"][cls_id] += 1
                                print(f"🎋 [打标成功] 侧向捕获：【{label_name}】")

                self.latest_yolo_cx, self.latest_yolo_cls = -1, -1
                has_obs = False
                if self.real_lidar_data:
                    for dist in self.real_lidar_data:
                        if 0.05 < dist < 1.0: has_obs = True; break
                self.obstacle_in_1m = has_obs
                time.sleep(0.05)
            except: time.sleep(0.05)

    def get_current_telemetry_snapshot(self):
        obs_list = []
        MAX_UPLOAD_DIST = 4.0  
        if self.show_lidar_points and self.real_lidar_data:
            for i, distance in enumerate(self.real_lidar_data):
                if self.lidar_min_dist < distance < MAX_UPLOAD_DIST:
                    angle = self.angle_min + (i * self.angle_increment)
                    lx, ly = distance * math.cos(angle), distance * math.sin(angle) * (-1)
                    wx = self.car_world_x + (lx * math.cos(self.car_yaw) - ly * math.sin(self.car_yaw))
                    wy = self.car_world_y + (lx * math.sin(self.car_yaw) + ly * math.cos(self.car_yaw))
                    obs_list.append([wx, wy])
        
        if time.time() < getattr(self, "map_clear_force_empty_until", 0.0):
            clean_markers = []
        else:
            with semantic_lock:
                clean_markers = [m[:6] for m in self.semantic_markers]

        return {
            "car_world_x": self.car_world_x, "car_world_y": self.car_world_y, "car_yaw": self.car_yaw, 
            "obstacle_in_1m": self.obstacle_in_1m, 
            "recorded_path": list(self.path_log), 
            "permanent_obstacles": obs_list, "semantic_markers": clean_markers, "map_clear_seq": self.map_clear_seq,
            "ai_running": self.task_running, "ai_counts": self.stats["counts"][:7],
            "battery_voltage": self.real_battery_vcc, "min_dist": self.lidar_min_dist, "max_dist": self.lidar_max_dist, 
            "show_lidar_points": self.show_lidar_points, "spray_master_switch": self.spray_master_switch,
            "f1_soil_data": self.latest_soil_packet  
        }

    def start_task(self):
        if self.task_running: return
        self.macro_queue.clear()
        self.path_log.clear() 
        with semantic_lock:
            self.semantic_markers.clear()
            self.sprayed_weed_points.clear()
        self.map_clear_seq += 1
        self.map_clear_force_empty_until = time.time() + 0.5
        self.is_spraying_blocking = False
        self.was_auto_driving_before_spray = False
        self.current_action = "STOP"
        self.action_start_x, self.action_start_y, self.action_start_yaw = self.car_world_x, self.car_world_y, self.car_yaw
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.SAVE_DIR = os.path.join(self.BASE_DIR, f"Task_{self.timestamp}"); os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.stats["counts"] = [0] * 12 
        self.stats["details"] = []
        
        self.raw_video_writer = cv2.VideoWriter(os.path.join(self.SAVE_DIR, f"Video_Raw_{self.timestamp}.mp4"), self.fourcc, 20.0, (640, 480))
        self.box_video_writer = cv2.VideoWriter(os.path.join(self.SAVE_DIR, f"Video_Box_{self.timestamp}.mp4"), self.fourcc, 20.0, (640, 480))
        self.task_running = True
        print("🎬 ─── [端云一体智算生产链路已点火] ───")

    def stop_task(self):
        if not self.task_running: return
        self.change_macro_action("STOP"); self.task_running = False; time.sleep(0.3)
        self.was_auto_driving_before_spray = False
        if self.raw_video_writer: self.raw_video_writer.release(); self.raw_video_writer = None
        if self.box_video_writer: self.box_video_writer.release(); self.box_video_writer = None
        threading.Thread(target=self.finish_task_and_report, daemon=True).start()

    def finish_task_and_report(self):
        report_name = f"Report_{self.timestamp}.txt"
        report_path = os.path.join(self.SAVE_DIR, report_name)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"==================================================\n")
            f.write(f" 智能农业巡检机器人 - 任务执行综合报告\n")
            f.write(f"==================================================\n")
            f.write(f"任务唯一ID : {self.timestamp}\n")
            f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"累计标绘巡检植物总点数: {len(self.semantic_markers)} 个\n\n")
            
            f.write(f"┌── [ 终端物理层：全域土壤肥力断点采样 ] ──────────────────┐\n")
            if self.latest_soil_packet:
                f.write(f"│  采样时间戳 : {self.latest_soil_packet.get('time_str', '未知')}\n")
                f.write(f"│  空间绝对坐标: X = {round(self.car_world_x, 3)} m, Y = {round(self.car_world_y, 3)} m\n")
                f.write(f"│  ──────────────────────────────────────────────\n")
                f.write(f"│  土壤温度   : {self.latest_soil_packet.get('temp', 0.0)} ℃\n")
                f.write(f"│  土壤水分   : {self.latest_soil_packet.get('moisture', 0.0)} %\n")
                f.write(f"│  有效氮 (N) : {self.latest_soil_packet.get('nitrogen', 0)} mg/kg\n")
                f.write(f"│  有效磷 (P) : {self.latest_soil_packet.get('phosphorus', 0)} mg/kg\n")
                f.write(f"│  有效钾 (K) : {self.latest_soil_packet.get('potassium', 0)} mg/kg\n")
            else:
                f.write(f"│  当前结束位置: X = {round(self.car_world_x, 3)} m, Y = {round(self.car_world_y, 3)} m\n")
                f.write(f"│  提示: 任务期间未接收到有效的土壤传感器数据包。\n")
            f.write(f"└────────────────────────────────────────────────────────┘\n\n")
            
            f.write(f"┌── [ 视觉感知层：高精空间语义标绘清单 ] ──────────────────┐\n")
            if self.stats["details"]:
                f.write(f"│  {'序号':<4} | {'触发时间':<19} | {'目标类别':<8} | {'相对车体方位':<10} | {'世界坐标 (X, Y)'}\n")
                f.write(f"│  ───────────────────────────────────────────────────────\n")
                f.write(f"│  \n")
                for idx, item in enumerate(self.stats["details"], 1):
                    f.write(f"│  {idx:<4} | {item['time']} | {item['label']:<8} | {item['side']:<10} | ({item['x']:.3f}, {item['y']:.3f}) m\n")
            else:
                f.write(f"│  本次巡检任务未标绘到满足过滤条件的杂草或作物目标。\n")
            f.write(f"└────────────────────────────────────────────────────────┘\n")
            
        try: 
            subprocess.Popen([sys.executable, "brain_report.py", report_path, self.PC_IP, self.SAVE_DIR, self.timestamp])
            subprocess.Popen([sys.executable, "media_uploader.py", report_path, self.PC_IP, self.SAVE_DIR, self.timestamp])
            print(f"📊 [智算生产链路] 数据生成就绪，跨网段盲传管道并联拉起。")
        except: pass


if __name__ == "__main__":
    rclpy.init(args=None)
    master_service = AgriRobotHeadlessMaster()
    ros_node = RosGameReceiver(master_service)
    threading.Thread(target=lambda: rclpy.spin(ros_node), daemon=True).start()

    print("🧠 正在初始化 RK3588 边缘计算 NPU 硬件加速核心...")
    pool = rknnPoolExecutor(rknnModel="./rknnModel/best.rknn", TPEs=3, func=myFunc)
    print("✅ RKNN 边缘智算推理线程池组装完毕！")

    cap = cv2.VideoCapture("/dev/video21")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    latest_frame = None; frame_id = 0; last_processed_id = -1; frame_lock = threading.Lock(); camera_ready = False

    def camera_reader_worker():
        global latest_frame, camera_ready, frame_id
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                camera_ready = True
                with frame_lock: latest_frame = frame; frame_id += 1  
            else: time.sleep(0.005)

    threading.Thread(target=camera_reader_worker, daemon=True).start()
    while not camera_ready: time.sleep(0.05)
    master_service.safe_serial_send(bytes.fromhex("AA 2C 01"))

    try:
        sys.path.append("/home/elf")  
        import aotu
        if hasattr(aotu, 'TxtRouteExecutor'):
            master_service.route_executor = aotu.TxtRouteExecutor(master_service, default_route_path=master_service.DEFAULT_ROUTE_PATH)
            master_service.route_executor.start_worker()
            print("🧭 [系统核心组件] 自动驾驶状态机组件已挂载！")
    except Exception as e: print(f"❌ [严重异常] 状态机后置注入失败: {e}")

    try:
        while cap.isOpened():
            with frame_lock:
                if latest_frame is None or frame_id == last_processed_id: time.sleep(0.002); continue
                frame = latest_frame.copy(); last_processed_id = frame_id 
                                    
            if master_service.task_running and master_service.raw_video_writer:
                master_service.raw_video_writer.write(frame)
                
            pool.put(frame); result, flag = pool.get()
            if flag and isinstance(result, dict):
                frame_res = result["img"]
                if master_service.task_running and master_service.box_video_writer:
                    master_service.box_video_writer.write(frame_res)

                detections = result.get("detections", [])
                if detections: cx, cls_id = master_service.select_center_target_from_detections(detections)
                else:
                    coords = result.get("coords", [])
                    cls_id = result.get("class_id", -1)
                    cx = int(coords[0]) if (len(coords) == 2 and master_service.target_zone_min <= int(coords[0]) <= master_service.target_zone_max) else -1

                master_service.latest_yolo_cx = cx
                master_service.latest_yolo_cls = cls_id
                frame_to_send = frame_res
            else: frame_to_send = frame

            if master_service.compound_asset_queue.full():
                try: master_service.compound_asset_queue.get_nowait()
                except queue.Empty: pass
            
            telemetry_data = master_service.get_current_telemetry_snapshot()
            asset_pack = (frame_to_send.copy(), telemetry_data)
            try: master_service.compound_asset_queue.put_nowait(asset_pack)
            except queue.Full: pass
            time.sleep(0.005)
                                    
    except KeyboardInterrupt: pass
    finally: master_service.stop_task(); cap.release(); pool.release()
