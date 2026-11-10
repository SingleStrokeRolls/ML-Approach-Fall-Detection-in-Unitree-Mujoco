from pathlib import Path
TASK_DIR = Path(__file__).resolve().parent

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

# Path to your XML file (change if needed)
XML_PATH = str(TASK_DIR / "cushioning_test.xml")
CSV_NAME = TASK_DIR / 'acc_with_cushion.csv'

# Export CSVs
def export_csv(list):
    df = pd.DataFrame(list, columns=['t','x', 'y', 'z'])
    df.to_csv(CSV_NAME, index=False)

# Load the model and create data
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

model.opt.timestep = 0.002  # 2ms

acc = []
t = []

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

    print("Viewer running. Press ESC or close window to exit.")
    
    while viewer.is_running():
        # Step the simulation forward (optional - comment out for static view)
        mujoco.mj_step(model, data)
        
        acc.append(data.sensordata[0:3].copy())
        t.append(round(data.time,3))

        # Sync the viewer with the current physics state
        viewer.sync()

        # End Sim
        if data.time > 2:
            print(f"### {max((row[2] for row in acc), key=abs)} ###")
            print(f"### {min((row[2] for row in acc), key=abs)} ###")
            combined = [[t, a[0], a[1], a[2]] for t, a in zip(t, acc)]
            export_csv(combined)
            break