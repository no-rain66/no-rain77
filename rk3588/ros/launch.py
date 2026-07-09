import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 获取当前功能包的 share 目录路径，用于定位 config 文件夹
    robot_driver_share_dir = get_package_share_directory('robot_driver')

    # ================= 1. 自动给激光雷达串口赋权限 =================
    set_permissions = ExecuteProcess(
        cmd=['sudo', 'chmod', '777', '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyS9'],
        shell=True
    )

    # ================= 2. 引入雷达驱动的 Launch =================
    rplidar_share_dir = get_package_share_directory('rplidar_ros')
    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rplidar_share_dir, 'launch', 'rplidar_c1_launch.py')
        )
    )

    # ================= 3. 启动小车底盘节点 =================
    robot_node = Node(
        package='robot_driver',
        executable='robot_node',
        name='robot_node',
        output='screen'
    )

    # ================= 🌟 核心新增：启动官方扩展卡尔曼滤波节点 (EKF) =================
    # 读取我们刚才建立的原 config/ekf.yaml
    ekf_config_path = os.path.join(robot_driver_share_dir, 'config', 'ekf.yaml')
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path]
    )

    # ================= 4. 启动雷达数据过滤器脚本 =================
    filter_script_path = os.path.expanduser('~/ros2_ws/src/filter.py')
    filter_node = ExecuteProcess(
        cmd=['python3', filter_script_path],
        output='screen'
    )

    # ================= 5. 启动静态坐标变换 TF (恢复为标准 0 度朝向) =================
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'laser'],
        output='screen'
    )

    # ================= 🎯 6. 一键融合纯雷达高精建图节点 =================
    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        # 传递专属纯雷达配置文件的路径及文件名
        arguments=[
            '-configuration_directory', os.path.expanduser('~/ros2_ws/src/robot_driver/launch'),
            '-configuration_basename', 'agri_slam.lua'
        ],
        # 优雅话题重映射：强行把默认消费的 /scan 话题改吃你的净化话题 /scan_filtered
        # 🎯 核心修改 1：让建图直接订阅底盘送出来的原始里程计 /odom_raw
        remappings=[
            ('scan', '/scan_filtered'),
            ('odom', '/odom_raw')
        ]
    )

    # ================= 返回所有要启动的节点和配置 (实现一键大一统全家桶) =================
    return LaunchDescription([
        set_permissions,
        robot_node,
        # ekf_node,           # 🎯 核心修改 2：加井号注销它，暂时不用卡尔曼滤波融合
        rplidar_launch,
        filter_node,
        static_tf_node,
        cartographer_node
    ])
