import mujoco
import numpy as np
import os

def control_joint_to_angle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    desired_qpos: float,
    kp: float = 400.0,
    kd: float = 40.0,
    max_torque: float = 200.0,
    feedforward_torque: float = 0.0,
    use_actuator_limits: bool = True
) -> float:
    """
    Simple PD controller that drives one joint toward a desired position (qpos).
    
    Returns the computed control value (torque / effort) to be set in data.ctrl.
    
    Most common usage pattern:
        ctrl_value = control_joint_to_angle(model, data, "shoulder_pitch", -0.8)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_pitch_motor")
        data.ctrl[actuator_id] = ctrl_value
    """
    # ─── Get joint ID ───────────────────────────────────────────────
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jnt_id < 0:
        raise ValueError(f"Joint '{joint_name}' not found in model")

    # ─── Current state ──────────────────────────────────────────────
    current_qpos = data.qpos[jnt_id]
    current_qvel = data.qvel[jnt_id]

    # ─── PD law ─────────────────────────────────────────────────────
    position_error = desired_qpos - current_qpos
    velocity_error = 0.0 - current_qvel          # usually want to stop at target

    torque = (
        kp * position_error +
        kd * velocity_error +
        feedforward_torque
    )

    # ─── Torque limiting ────────────────────────────────────────────
    if use_actuator_limits:
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        ctrlrange = model.actuator_ctrlrange[act_id]
        torque = np.clip(torque, ctrlrange[0], ctrlrange[1])

    return torque

def export_csv(csv_name, t_store, gyro_store, acc_store, tof_store):
    fullpath = os.fspath(csv_name)
    os.makedirs(os.path.dirname(os.path.abspath(fullpath)), exist_ok=True)
    
    # 4. 转 numpy
    t = np.array(t_store)
    gyro = np.array(gyro_store)
    acc = np.array(acc_store)
    tof = np.array(tof_store)
    
    data = np.column_stack([t, acc, gyro, tof])
    # 保存为 CSV，带表头
    np.savetxt(
        fullpath,
        data,
        delimiter=',',
        fmt='%.6f', # 根据精度需求调整：%.3f, %.6f 等
        header='t,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,tof',
        comments='' # 去掉 # 注释符号，让表头干净
    )
    print(f"### {csv_name} saved. ###")