#!/usr/bin/env python3
import time
import math
import threading

class TxtRouteExecutor:
    def __init__(self, master_robot, default_route_path):
        """
        🚜 甘蔗地全自动蛇形巡航轨迹规划状态机
        👉 核心重构：彻底移除了顶部对 main.py 的 import，通过参数直接持有外部 master 实例句柄
        """
        self.master = master_robot  # 动态注入的下位机核心控制类（AgriRobotHeadlessMaster 实例）
        self.default_route_path = default_route_path
        
        # 🗺️ 巡航运动物理核心参数（地面站动态修改的目标内存）
        self.row_length = 6.0       # 单垄行长（米）
        self.row_spacing = 0.70     # 垄间行距（米）
        self.total_rows = 4         # 作业总行数
        self.turn_dir = 1           # 1: 右转优先，-1: 左转优先
        
        # 🧭 状态机内部运行状态统计总线
        self.current_row_index = 0  # 当前正在作业的行号 (0 代表第1行)
        self.sub_state = "IDLE"     # 内部子状态机: IDLE(待机), TRACKING(垄间巡航), TURNING(变垄调头), FINISHED(作业完毕)
        
        self._is_running = False    # 自动驾驶总线使能开关
        self.worker_thread = None

    def start_worker(self):
        """挂载状态机后台工人线程"""
        if self.worker_thread is None:
            self._is_running = False
            self.worker_thread = threading.Thread(target=self._main_route_loop, daemon=True)
            self.worker_thread.start()
            print("🧭 [aotu状态机] 异步闭环控制工人线程已挂载就绪。")

    def is_running(self):
        """供主控及地面站轮询的运行使能状态"""
        return self._is_running

    def snapshot(self):
        """
        📊 动态序列化当前状态机的快照包，塞入遥测包送往网页前端大盘显示
        """
        return {
            "auto_patrol_state": f"{self.sub_state} (第 {self.current_row_index + 1}/{self.total_rows} 垄)",
            "row_length": float(self.row_length),
            "row_spacing": float(self.row_spacing),
            "total_rows": int(self.total_rows),
            "turn_dir": int(self.turn_dir),
            "nav_running": bool(self._is_running)
        }

    def start(self, row_length, row_spacing, total_rows, turn_dir):
        """
        🔥 响应地面站的点击“启动巡航”纯文本指令，点火自动驾驶
        """
        # 1. 实时拉取并覆写物理大盘传入的最新数值
        self.row_length = row_length
        self.row_spacing = row_spacing
        self.total_rows = total_rows
        self.turn_dir = turn_dir
        
        print("\n" + "="*50)
        print(f"🚜 [aotu驾驶激活] 自主闭环规划状态机点火成功！")
        print(f"📐 作业规划指标 => 单垄行长: {self.row_length:.2f}米 | 垄间间距: {self.row_spacing:.2f}米")
        print(f"🔢 覆盖密度指标 => 总覆盖垄数: {self.total_rows}行 | 调头首选方向: {'右转优先' if self.turn_dir > 0 else '左转优先'}")
        print("="*50 + "\n")

        # 2. 状态指针初始化
        self.current_row_index = 0
        self.sub_state = "TRACKING"
        self._is_running = True
        
        # 3. 自动联动主程序的任务录制功能（开启录像与时序土壤归档资产流水线）
        if hasattr(self.master, "start_task"):
            self.master.start_task()

    def stop(self, finish_report=False, reason=""):
        """安全斩断截断自动巡航"""
        if not self._is_running: return
        self._is_running = False
        self.sub_state = "IDLE"
        
        # 物理刹车下发
        if self.master:
            self.master.safe_serial_send(bytes.fromhex("AA 07 01")) # 发送底层急停 HEX 帧
        
        print(f"🛑 [aotu驾驶强切] 自动规划状态机被安全挂起。阻断触发源: {reason}")
        
        if finish_report and hasattr(self.master, "stop_task"):
            self.master.stop_task()

    def _main_route_loop(self):
        """
        🚀 状态机核心控制循环（完全与 main.py 主推理流水线解耦并列运行）
        """
        # 建立局部变量缓存位置坐标，用来计算相对位移
        start_x, start_y = 0.0, 0.0
        is_init_step = True

        while True:
            try:
                if not self._is_running:
                    time.sleep(0.1)
                    continue

                # 🛡️ 靶向喷药断路器联锁：如果 main.py 正在执行倒车/驻车喷药，状态机交出底盘控制权，原地挂起等待
                if getattr(self.master, "is_spraying_blocking", False):
                    time.sleep(0.05)
                    continue

                # 🛰️ 提取当前小车的 ROS2 Odom 高精融合里程计位姿
                current_x = getattr(self.master, "car_world_x", 0.0)
                current_y = getattr(self.master, "car_world_y", 0.0)
                current_yaw = getattr(self.master, "car_yaw", 0.0)

                # ==============================================================================
                # 🧭 子状态机分支 1：垄间直道高精巡航模式
                # ==============================================================================
                if self.sub_state == "TRACKING":
                    if is_init_step:
                        start_x, start_y = current_x, current_y
                        is_init_step = False
                        print(f"🌱 [状态机控制] 正在切入第 {self.current_row_index + 1} 垄，直道原位感知开启。")

                    # 计算小车沿着直道方向行驶的累计欧氏位移
                    distance_traveled = math.hypot(current_x - start_x, current_y - start_y)

                    if distance_traveled < self.row_length:
                        # 闭环居中修正：自主下发前进 HEX，交由 main.py 的雷达间壁测距去动态调节左右偏摆
                        if self.master:
                            self.master.safe_serial_send(bytes.fromhex("AA 02 01")) # 持续前行
                    else:
                        # 当前垄作业结束，判断是否需要变垄调头
                        print(f"🏁 [状态机控制] 第 {self.current_row_index + 1} 垄作业完毕，累计行驶: {distance_traveled:.2f} 米。")
                        if self.current_row_index + 1 < self.total_rows:
                            self.current_row_index += 1
                            self.sub_state = "TURNING"
                            is_init_step = True
                        else:
                            self.sub_state = "FINISHED"

                # ==============================================================================
                # 🧭 子状态机分支 2：垄尾 U 型变垄调头模式
                # ==============================================================================
                elif self.sub_state == "TURNING":
                    # 计算当前应当转向的角度目标（以右转优先或左转优先进行 180° 摆头）
                    # 此处根据你的变垄半径和行距进行硬件级别的原地转向控制
                    print(f"🔄 [状态机控制] 正在垄尾执行调头调平变垄，准备并轨进入第 {self.current_row_index + 1} 垄...")
                    
                    # 模拟底层变垄调头控制：此处根据具体差速小车底盘特性进行串口发送
                    # 实际生产中可结合里程计绝对航向角 yaw 进行精准 180 度大回旋闭环
                    if self.master:
                        # 根据 turn_dir 下发左转弯或右转弯 hex 序列
                        turn_cmd = "AA 08 01" if self.turn_dir > 0 else "AA 09 01"
                        self.master.safe_serial_send(bytes.fromhex(turn_cmd))
                    
                    time.sleep(2.5) # 预留原地回旋调头物理耗时（可根据实际底盘惯性重置）
                    
                    # 调头完成后，状态切回直道追踪
                    self.sub_state = "TRACKING"
                    is_init_step = True

                # ==============================================================================
                # 🧭 子状态机分支 3：全田块闭环全景作业安全收车
                # ==============================================================================
                elif self.sub_state == "FINISHED":
                    print("🎉 [自动巡航系统] 恭喜！当前规划的全部高精数字地块已 100% 覆盖巡检完毕，正在触发安全收车保护...")
                    self.stop(finish_report=True, reason="ROUTE_TASK_COMPLETED")

                time.sleep(0.05) # 20Hz 频率平抑边缘端 CPU 的轮询压力

            except Exception as e:
                print(f"⚠️ [aotu状态机异常线程守卫]: {e}")
                time.sleep(0.1)
