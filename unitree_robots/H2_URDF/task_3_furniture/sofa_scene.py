from pathlib import Path
import sys

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR.parent / "shared"))

import mujoco
import mujoco.viewer
import numpy as np
import time
from starter_helper import export_csv

def main():
    # Path to your XML file (change if needed)
    XML_PATH = str(TASK_DIR / "sofa_scene.xml")
    SAVED_FILE_NAME = "sofa_sitting.csv"
    EXPORT = False

    # Load the model and create data
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    model.opt.timestep = 0.002  # 2ms

    # Data Storage
    acc_store = []
    gyro_store = []
    tof_store = []
    t_store = []
    qvel_store = []

    # Get Joint Id
    right_hip_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_pitch_joint")
    left_hip_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch_joint")
    waist_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")
    right_knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee_joint")
    left_knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_joint")

    # Get Motor Id
    right_hip_pitch_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip_pitch_joint")
    left_hip_pitch_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hip_pitch_joint")
    waist_pitch_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "waist_pitch_joint")
    right_knee_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_knee_joint")
    left_knee_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_joint")

    # Launch the interactive viewer (most common & simplest way)
    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui   = True,   # show simulation controls on left
        show_right_ui  = True,   # show visualization options on right
    ) as viewer:
        
        # Optional: customize viewer at start
        viewer.cam.azimuth   = 120     # horizontal camera angle
        viewer.cam.elevation = -20     # vertical camera angle
        viewer.cam.distance  = 8.0     # zoom level
        viewer.cam.lookat    = [0, 0, 0.5]  # look at center-ish of scene

        print("Viewer running. ")
        
        while viewer.is_running():

            if 0 <= data.time <= 0.002:
                # xfrc_applied[body_id, 0:3] 是力，[3:6] 是力矩
                # data.xfrc_applied[torso_link_body_id, :3] = [0,500,0]
                # Hip Joint Starting Position
                data.qpos[waist_pitch_id+6] = 0
                data.qpos[right_hip_pitch_id+6] = -0.25
                data.qpos[left_hip_pitch_id+6] = -0.25
                data.qpos[left_knee_id+6] = 0.15
                data.qpos[right_knee_id+6] = 0.15
                mujoco.mj_forward(model, data)

            # Control
            # data.ctrl[waist_pitch_motor_id] = -5
            # data.ctrl[waist_pitch_motor_id] = data.time * 10
            # data.ctrl[left_hip_pitch_motor_id] = data.time * 18
            # data.ctrl[right_hip_pitch_motor_id] = data.time * 18
            # if data.time <= 0.4:
            #     data.ctrl[left_knee_motor_id] = data.time * -8.8
            #     data.ctrl[right_knee_motor_id] = data.time * -8.8

            mujoco.mj_step(model, data)

            # Record data
            acc_store.append(data.sensor("imu_acc").data.copy())
            gyro_store.append(data.sensor("imu_gyro").data.copy())
            tof_store.append(data.sensor("tof").data.copy())
            qvel_store.append([data.qvel[right_hip_pitch_id+6],data.qvel[left_hip_pitch_id+6]])
            t_store.append(data.time)
            # print(data.time, data.sensor("imu_acc").data.copy())

            # Sync the viewer with the current physics state
            viewer.sync()

            # time.sleep(0.001)

        # Export CSV
        # quickAccExport(t_store, qvel_store)
        if EXPORT:
            export_csv(TASK_DIR / "csv" / SAVED_FILE_NAME, t_store, gyro_store, acc_store, tof_store)

def print_all_joints(model: mujoco.MjModel):
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"(unnamed_{i})"
        print(f"{i:3d}  {name}")

def quickAccExport(t_store, qvel):
    fullpath = str(TASK_DIR / 'csv/qvel.csv')
    data = np.column_stack([t_store,qvel])
    # 保存为 CSV，带表头
    np.savetxt(
        fullpath,
        data,
        delimiter=',',
        fmt='%.6f', # 根据精度需求调整：%.3f, %.6f 等
        header='t,qvel_right_hip_pitch,qvel_left_hip_pitch',
        comments='' # 去掉 # 注释符号，让表头干净
    )
    print("Quick Exported.")

if __name__ == "__main__":
    main()