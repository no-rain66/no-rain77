# -*- coding: utf-8 -*-
import sys
import os
import time
import subprocess
import requests

def main():
    if len(sys.argv) < 5:
        print("❌ 参数捕获链条断裂，无法执行接力。")
        return

    raw_report_path = sys.argv[1]
    pc_ip = sys.argv[2]
    save_dir = sys.argv[3]
    timestamp = sys.argv[4]

    print(f"\n⚡ [{time.strftime('%X')}] 独立大模型接力进程成功脱离主程序包袱，单独启动！")
    print("⏳ 正在进入系统硬件安全冷却期（死等 5.0 秒，保障内核完全回收 NPU 显存）...")
    time.sleep(5.0)  # 🧱 给瑞芯微驱动异步释放 YOLO 显存的生命线

    # 读取前线原始报告流水账
    try:
        with open(raw_report_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        print(f"❌ 读取前线原始报告失败: {e}")
        return

    # 🪐 构造投喂给纯文本大模型交互命令的 Prompt
    prompt = (
        "你是一个顶级的自动化农业机器人诊断专家。请仔细阅读并分析以下小车在前线巡检采集到的第一手原始数据，"
        "并结合 YOLO 统计计数进行深度智能评估，为农户生成一份结构清晰、语言专业、高情商的农业质检大报。"
        "报告必须严格包含以下几大板块：\n"
        "1. 甘蔗苗情与生长势头评估\n"
        "2. 野生杂草危害等级评定与空间分布点数分析\n"
        "3. 接下来一到两周精确水肥追施、高效定向除草剂喷洒的专家养护建议。\n"
        "要求：请使用纯正、得体的中文字符直接输出整篇报告，不要带有任何多余的交互行或英文调试杂质。\n\n"
        "【巡检前线原始感知数据流如下】:\n" + raw_text
    )

    print("🚀 正在通过无图像纯文本模式唤醒本地大模型，全力推进后方多模态质检核查...")
    
    # 🎯 纯文本调用命令与工作路径对齐
    demo_cwd = "/home/elf/Data/model/demo_Linux_aarch64/"
    cmd = ["./llm", "../Qwen2-VL-2B-Instruct.rkllm", "512", "2048"]
    
    try:
        # 👑 【终极环境修复】：强行把 ./lib 注入子进程的系统环境变量，彻底破除动态库缺失的隐形暴毙！
        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = f"./lib:{my_env.get('LD_LIBRARY_PATH', '')}"

        # 随拉随走，纯文本交互不吃资源，点火速度极快
        process = subprocess.Popen(
            cmd, cwd=demo_cwd, env=my_env, # 🪐 成功注入生存环境环境
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', bufsize=1
        )
        
        # 纯文本处理速度极高，直接通过 communicate 注入 Prompt 并静等它回归退出
        # 给足大模型 45 秒自回归吐字的真空爆发期
        stdout, stderr = process.communicate(input=prompt, timeout=45)
        
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        print("⚠️ 大模型自回归推理超时，为保护硬件执行强制熔断。")
    except Exception as e:
        print(f"❌ 调用大模型交互可执行程序发生异常: {e}")
        return

    # 清洗大模型输出的标准流
    expert_analysis = stdout.strip()
    
    # 过滤掉由于底层库打印可能残留在首尾的版本调试行
    clean_lines = []
    for line in expert_analysis.split('\n'):
        if "rkllm" not in line.lower() and "rknn" not in line.lower() and "main:" not in line.lower() and "user:" not in line.lower() and "robot:" not in line.lower():
            clean_lines.append(line)
    
    final_analysis_text = "\n".join(clean_lines).strip()
    
    if not final_analysis_text:
        print("❌ NPU 管道数据打捞失败，标准输出中未检测到生成汉字。")
        print("底层错误信息:", stderr)
        final_analysis_text = "【系统异常】本地常驻大模型未能成功返回纯净化专家中文诊断，请检查板卡底层驱动版本对齐状态。"

    # 📝 3. 固化拼装最终的大模型智能润色核查大报
    expert_report_name = f"Expert_Report_{timestamp}.txt"
    expert_report_path = os.path.join(save_dir, expert_report_name)
    
    content = "==================================================\n"
    content += f" 🌾 Qwen2-VL 本地AI智库 - 农田精细化巡检专家智能大报\n"
    content += "==================================================\n"
    content += f"核查任务ID: {timestamp}\n"
    content += f"报告固化生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    content += "--------------------------------------------------\n"
    content += final_analysis_text + "\n"
    content += "==================================================\n"

    with open(expert_report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✨ 完美融合版专家级智能分析大报已成功本地固化: {expert_report_path}")

    # ☁️ 4. 通过跨网段推送至电脑端暂存区
    cloud_api = f"http://{pc_ip}:5000/api/upload-report" 
    try:
        with open(expert_report_path, 'rb') as f:
            requests.post(cloud_api, files={'file': f}, timeout=15)
            print(f"☁️ 跨网段网络动脉打通！高级大模型大报已成功秒传至远程控制台暂存区: {pc_ip}")
    except Exception as e:
        print(f"⚠️ 报告推送远端失败（请检查接收端后端服务或网络通路）: {e}")

if __name__ == "__main__":
    main()
