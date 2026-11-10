# Humanoid Fall Detection and Response Analysis in MuJoCo

Three simulation tasks connect robot experiments with recorded sensor data and analysis notebooks. The standing workflow uses H1-2; cushion and furniture experiments use H2.

| Task | Code and scenes | Analysis |
| --- | --- | --- |
| [1. Standing fall detection](simulate_python/task_1_standing/README.md) | `simulate_python/task_1_standing/`: collection, LSTM training, interactive/batch inference | Sensor visualization, training, detection timing, model export notebooks |
| [2. Directional cushion falls](unitree_robots/H2_URDF/task_2_cushion/README.md) | `unitree_robots/H2_URDF/task_2_cushion/`: backward, forward, leftward experiments | Sensor visualization and supporting material/contact experiment |
| [3. Chair and sofa interaction](unitree_robots/H2_URDF/task_3_furniture/README.md) | `unitree_robots/H2_URDF/task_3_furniture/`: seating/contact and chair disturbance | Sensor analysis, filtering, and comparison-recording notebook |

## Repository layout

```text
simulate_python/
  task_1_standing/          # Scripts, notebooks, saved models, and recorded datasets
unitree_robots/
  h1_2/                    # Standing robot model and meshes
  H2_URDF/
    shared/                # H2 model, meshes, and helper used by both H2 tasks
    task_2_cushion/         # Cushion scenes, runner, csv/, and analysis
      material_validation/ # Supporting ball/cushion experiment
    task_3_furniture/       # Chair/sofa scenes, runners, csv/, and analysis
```

## Getting started

Use a Python environment compatible with MuJoCo and TensorFlow, then install the analysis/simulation dependencies:

```sh
python -m pip install -r requirements.txt
```

Standing runners also require [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python) and its DDS dependencies. Configure the available network interface in `simulate_python/task_1_standing/config.py`. H2 runners use MuJoCo directly. Exact dependency versions have not yet been validated or pinned.

Run scripts from the repository root using the commands in each task guide. Open each task's notebooks in Jupyter or your IDE; their first cell selects the appropriate working directory. Existing notebook outputs were cleared to prevent moved notebooks from displaying stale results.

## Reading the results

Follow each task from its runner to CSV recordings to the named analysis notebook. The workflows are experimental: the standing notebook currently uses an episode that is also in training for an example inference check; some H2 motions directly set joint positions. These do not yet establish held-out detection accuracy or validated protective control. The tasks are separate experiments; the standing detector does not automatically trigger H2 responses.

## Scope and attribution

Built on the [Unitree MuJoCo simulator](https://github.com/unitreerobotics/unitree_mujoco), using Unitree robot assets and SDK integration. See [LICENSE](LICENSE). The original general-purpose simulator README was replaced with this task-oriented guide.

This portfolio layout retains the standing workflow, H2 experiments, relevant recorded data, and supporting cushion test. Unrelated robot families, C++/ROS examples, terrain tools, general simulator launchers/tests, unused H2 model variants, fusion experiments, and generated debug logs were removed. Original experiment filenames are retained inside task folders to make the migration traceable in Git. Dataset contents and existing model binaries were preserved; this reorganization does not retrain models or alter the experimental labeling scheme.
