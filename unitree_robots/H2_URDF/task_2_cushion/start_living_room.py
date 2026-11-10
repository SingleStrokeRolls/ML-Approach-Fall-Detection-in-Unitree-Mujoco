from pathlib import Path
import sys

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR.parent / "shared"))

import mujoco
import mujoco.viewer
import numpy as np
from starter_helper import *

# Path to XML file
BACKWARD_PATH = str(TASK_DIR / "scene_with_cushion.xml")
FORWARD_PATH = str(TASK_DIR / "scene_with_cushion_forward.xml")
LEFTWARD_PATH = str(TASK_DIR / "scene_with_cushion_leftward.xml")

FILE_NAME = "0413_leftward_with_arms.csv"

SCENARIO_LIST = ["BACKWARD", "BACKWARD_TWIST",
                 "FORWARD_FREE_FALL","FORWARD_WITH_ARMS",
                 "LEFTWARD_FREE_FALL","LEFTWARD_WITH_ARMS"]
SCENARIO = "LEFTWARD_WITH_ARMS"

# Private Define
AUTO_END = False
SAVE = True
WAIT_THRESHOLD = 1 # second
TimeCounter = 0

# Load the model and create data
SCENE_PATH = (BACKWARD_PATH if SCENARIO.startswith("BACKWARD") else
              FORWARD_PATH if SCENARIO.startswith("FORWARD") else LEFTWARD_PATH)
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data  = mujoco.MjData(model)

# Data Initialization
t_store = []
gyro_store = []
acc_store = []
tof_store = []
 
# Get Knee Motor Id
left_knee_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_joint")
right_knee_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_knee_joint")
# Get Waist Joint Id and Initial QPOS
waist_yaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "waist_yaw_joint")
waist_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "waist_roll_joint")
waist_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")
init_yaw_qpos = data.qpos[waist_yaw_id].copy()
init_roll_qpos = data.qpos[waist_roll_id].copy()
init_pitch_qpos = data.qpos[waist_pitch_id].copy()
# Get Waist Motor Id
waist_yaw_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "waist_yaw_joint")
waist_roll_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "waist_roll_joint")
waist_pitch_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "waist_pitch_joint")
# Get Torso Body Id
torso_link_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
# Get Hip Pitch Joint and Motor Id
right_hip_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_pitch_joint")
left_hip_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch_joint")
right_hip_pitch_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip_pitch_joint")
left_hip_pitch_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hip_pitch_joint")
# Get Hip Roll Joint Id
right_hip_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_roll_joint")
left_hip_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_roll_joint")
# Get Knee and Ankle Pitch Joint
right_knee_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee_joint")
left_knee_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_joint")
right_ankle_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_ankle_pitch_joint")
left_ankle_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_ankle_pitch_joint")
# Get Left/Right Shoulder Motor Id
left_shoulder_pitch_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pitch_joint")
right_shoulder_pitch_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch_joint")
left_shoulder_roll_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_roll_joint")
right_shoulder_roll_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_roll_joint")
# Get Left/Right Shoulder Joint Id
left_shoulder_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pitch_joint")
right_shoulder_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch_joint")
left_shoulder_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_roll_joint")
right_shoulder_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_roll_joint")

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
    viewer.cam.lookat    = [-1.5, 0, 0.5]  # look at center-ish of scene

    print("Viewer running. ")
    
    while viewer.is_running():
        # Reset Time Counter when reset Sim
        if data.time == 0 and TimeCounter > 1.004:
            TimeCounter = 0

        match(SCENARIO):
            case "BACKWARD":
                # Initial Pos
                if 0 <= data.time <= 0.002:
                    # xfrc_applied[body_id, 0:3] 是力，[3:6] 是力矩
                    # data.xfrc_applied[torso_link_body_id, :3] = [0,500,0]
                    # Hip Joint Starting Position
                    data.qpos[waist_pitch_id+6] = -0.2
                    data.qpos[right_hip_pitch_id+6] = -0.25
                    data.qpos[left_hip_pitch_id+6] = -0.25
                    mujoco.mj_forward(model, data)
                # Control
                # if data.time < 0.05:
                #     data.ctrl[right_knee_id] = data.ctrl[left_knee_id] = 5
                #     data.ctrl[waist_pitch_motor_id] = -15
                # else:
                #     data.ctrl[right_knee_id] = data.ctrl[left_knee_id] = 0
                #     data.ctrl[waist_pitch_motor_id] = 0

                # data.ctrl[waist_roll_motor_id] = control_joint_to_angle(model, data, "waist_roll_joint", init_roll_qpos)

            case "BACKWARD_TWIST":
                # Initial Pos
                if 0 <= data.time <= 0.002:
                    data.qpos[waist_pitch_id+6] = -0.2
                    data.qpos[right_hip_pitch_id+6] = -0.15
                    data.qpos[left_hip_pitch_id+6] = -0.15
                    mujoco.mj_forward(model, data)

                # External Force
                if 0 <= data.time <= 0.02:
                    data.xfrc_applied[torso_link_body_id, :3] = [0,250,0]

                # Control
                if 0 <= data.time <= 0.5:
                    data.ctrl[left_knee_motor_id] = 4
                    data.ctrl[right_knee_motor_id] = 4
                    data.ctrl[left_hip_pitch_motor_id] = -26
                    data.ctrl[right_hip_pitch_motor_id] = 32
                else:
                    data.ctrl[left_knee_motor_id] = 0
                    data.ctrl[right_knee_motor_id] = 0
                    data.ctrl[left_hip_pitch_motor_id] = 0
                    data.ctrl[right_hip_pitch_motor_id] = 0

                if 0 <= data.time <= 1.3:
                    data.qpos[waist_yaw_id+6] = data.time * 1

                data.qpos[left_hip_roll_id+6] = 0
                data.qpos[right_hip_roll_id+6] = 0

            case "FORWARD_FREE_FALL":
                # Forward Initial Pos left/right_knee_joint 0.2 left/right_ankle_pitch_joint -0.2
                if 0 <= data.time <= 0.002:
                    data.qpos[left_knee_joint_id+6] = 0.2
                    data.qpos[right_knee_joint_id+6] = 0.2
                    data.qpos[left_ankle_pitch_id+6] = -0.2
                    data.qpos[right_ankle_pitch_id+6] = -0.2
                # External Force
                # if 0 <= data.time <= 0.02:
                #     data.xfrc_applied[torso_link_body_id, :3] = [0,-250,0]

            case "FORWARD_WITH_ARMS":
                # Forward Initial Pos left/right_knee_joint 0.2 left/right_ankle_pitch_joint -0.2
                if 0 <= data.time <= 0.002:
                    data.qpos[left_ankle_pitch_id+6] = -0.2
                    data.qpos[right_ankle_pitch_id+6] = -0.2
                # External Force
                if 0 <= data.time <= 0.02:
                    data.xfrc_applied[torso_link_body_id, :3] = [0,-250,0]
                # Control
                if 0.01 <= data.time <= 1.5:
                    data.ctrl[left_shoulder_pitch_motor_id] = -32 * data.time * 1.2
                    data.ctrl[right_shoulder_pitch_motor_id] = -32 * data.time * 1.2
                    data.ctrl[left_shoulder_roll_motor_id] = 10 * data.time # positive outward
                    data.ctrl[right_shoulder_roll_motor_id] = -10 * data.time # negative outward
                    data.ctrl[left_hip_pitch_motor_id] = 10
                    data.ctrl[right_hip_pitch_motor_id] = 10
                    data.ctrl[left_knee_motor_id] = data.time * 8
                    data.ctrl[right_knee_motor_id] = data.time * 8
                else:
                    data.ctrl[left_shoulder_pitch_motor_id] = 0
                    data.ctrl[right_shoulder_pitch_motor_id] = 0
                    data.ctrl[left_shoulder_roll_motor_id] = 0
                    data.ctrl[right_shoulder_roll_motor_id] = 0
                    data.ctrl[left_hip_pitch_motor_id] = 0
                    data.ctrl[right_hip_pitch_motor_id] = 0
                    data.ctrl[left_knee_motor_id] = 0
                    data.ctrl[right_knee_motor_id] = 0

            case "LEFTWARD_FREE_FALL":
                # Initial Pos
                if 0 <= data.time <= 0.002:
                    data.qpos[left_hip_roll_id+6] = -0.2
                    data.qpos[right_hip_roll_id+6] = -0.2
            
            case "LEFTWARD_WITH_ARMS":
                # Initial Pos
                if 0 <= data.time <= 0.002:
                    data.qpos[left_hip_roll_id+6] = -0.2
                    data.qpos[right_hip_roll_id+6] = -0.2
                
                # Control
                if 0.002 <= data.time <= 0.3:
                    data.ctrl[left_knee_motor_id] = data.time * 100
                    data.ctrl[right_knee_motor_id] = data.time * 100
                else:
                    data.ctrl[left_knee_motor_id] = 0
                    data.ctrl[right_knee_motor_id] = 0
                if 0.3 <= data.time <= 1:
                    data.qpos[left_shoulder_roll_id+6] = (data.time-0.3) * 1.1
                    data.ctrl[right_shoulder_pitch_motor_id] = -25
                    data.ctrl[right_shoulder_roll_motor_id] = 10
                    data.ctrl[right_shoulder_roll_motor_id+2] = 12 # elbow
                else:
                    data.ctrl[right_shoulder_pitch_motor_id] = 0
                    data.ctrl[right_shoulder_roll_motor_id] = 0
                    data.ctrl[right_shoulder_roll_motor_id+2] = 0

            case _:
                print("Unknown Scenario.")

        # Step the simulation forward (optional - comment out for static view)
        if TimeCounter >= WAIT_THRESHOLD:
            mujoco.mj_step(model, data)
        TimeCounter = TimeCounter + 0.002

        # Logging
        # print(f'Pelvs Xpos: {data.xpos[pelvis_body_id]}')

        # Read Sensors
        # t_store.append(data.time)
        t_store.append(TimeCounter)
        gyro_store.append(data.sensor("imu_gyro").data.copy())
        acc_store.append(data.sensor("imu_acc").data.copy())
        tof_store.append(data.sensor("tof").data.copy())

        # Ending Condition
        if AUTO_END:
            if data.xpos[pelvis_body_id][2] < 0.061:
                break

        # Sync the viewer with the current physics state
        viewer.sync()

if SAVE:        
    export_csv(TASK_DIR / "csv" / FILE_NAME, t_store,gyro_store,acc_store, tof_store)