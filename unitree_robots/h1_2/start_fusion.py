import mujoco
import mujoco.viewer
import math
import numpy as np
import time

# Path to your XML file (change if needed)
XML_PATH = "scene.xml"

# Load the model and create imu_data
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

model.opt.timestep = 0.002
time.sleep(0.2)

def imu_process_imu_data(acc, gyro):
    return {
        'ax': acc[0] * 0.001 * 9.80665,
        'ay': acc[1] * 0.001 * 9.80665,
        'az': acc[2] * 0.001 * 9.80665,
        'gx': gyro[0] * 0.001,
        'gy': gyro[1] * 0.001,
        'gz': gyro[2] * 0.001
    }

def export_csv(csv_name, t, roll, pitch, yaw):

    # 4. 转 numpy
    t = np.array(t)
    roll = np.array(roll)
    pitch  = np.array(pitch)
    yaw = np.array(yaw)

    data = np.column_stack([t, roll, pitch, yaw])

    # 保存为 CSV，带表头
    np.savetxt(
        csv_name,
        data,
        delimiter=',',
        fmt='%.6f',           # 根据精度需求调整：%.3f, %.6f 等
        header='t,roll,pitch,yaw',
        comments=''           # 去掉 # 注释符号，让表头干净
    )

    print(f"CSV 已保存，共 {len(t)} 行数据")


# Launch the interactive viewer (most common & simplest way)
with mujoco.viewer.launch_passive(
    model,
    data,
    show_left_ui   = True,   # show simulation controls on left
    show_right_ui  = True,   # show visualization options on right
) as viewer:
    
    # Optional: customize viewer at start
    # viewer.cam.azimuth   = 120     # horizontal camera angle
    # viewer.cam.elevation = -20     # vertical camera angle
    # viewer.cam.distance  = 8.0     # zoom level
    # viewer.cam.lookat    = [0, 0, 0.5]  # look at center-ish of scene

    print("Viewer running. Press ESC or close window to exit.")

    _euler_state = {
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0
    }

    roll = []
    pitch = []
    yaw = []
    t_store = []

    dt = 0.002
    
    while viewer.is_running():
        # Step the simulation forward (optional - comment out for static view)
        mujoco.mj_step(model, data)

        gyro = data.sensor("imu_gyro").data.copy()
        acc = data.sensor("imu_acc").data.copy()
        imu_data = imu_process_imu_data(acc, gyro)
    
        _euler_state['roll'] += imu_data['gx'] * dt
        _euler_state['pitch'] += imu_data['gy'] * dt
        _euler_state['yaw'] += imu_data['gz'] * dt
        
        roll_acc = math.atan2(imu_data['ay'], imu_data['az']) * 57.29578
        pitch_acc = math.atan2(-imu_data['ax'], math.sqrt(imu_data['ay']**2 + imu_data['az']**2)) * 57.29578
        
        alpha = 0.98
        
        _euler_state['roll'] = alpha * _euler_state['roll'] + (1.0 - alpha) * roll_acc
        _euler_state['pitch'] = alpha * _euler_state['pitch'] + (1.0 - alpha) * pitch_acc

        roll.append(_euler_state['roll'])
        pitch.append(_euler_state['pitch'])
        yaw.append(_euler_state['yaw'])
        t_store.append(data.time)
        
        # Sync the viewer with the current physics state
        viewer.sync()
    export_csv("start_fusion_data.csv", t_store, roll, pitch, yaw)
    