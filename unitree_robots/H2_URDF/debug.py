import mujoco
import mujoco.viewer
import numpy as np

def main():
    # Path to your XML file (change if needed)
    XML_PATH = "scene_with_cushion.xml"

    # Load the model and create data
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    TimeCounter = 0
    flag = True

    # Get Hip Pitch Joint Id
    right_hip_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_pitch_joint")
    left_hip_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch_joint")
    waist_pitch_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "waist_pitch_joint")

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

            if flag:
                print_all_joints(model)
                flag = False

            if 0 <= data.time <= 0.002:    
                data.qpos[9] = -0.15 # Waist Pitch
                data.qpos[26] = -0.2 # Left Hip Pitch
                data.qpos[32] = -0.2 # Right Hip Pitch
                mujoco.mj_forward(model, data)

            if TimeCounter >= 2:
                mujoco.mj_step(model, data)
            TimeCounter = TimeCounter + 0.002
            # mujoco.mj_step(model, data)

            # Sync the viewer with the current physics state
            viewer.sync()

def print_all_joints(model: mujoco.MjModel):
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"(unnamed_{i})"
        print(f"{i:3d}  {name}")

if __name__ == "__main__":
    main()