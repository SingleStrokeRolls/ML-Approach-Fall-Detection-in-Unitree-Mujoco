# Task 3: Chair and sofa interactions

Question: how do furniture support and contact affect seated motion and falls?

## Trace the workflow

| Experiment | Runner | Scene | Recording |
| --- | --- | --- | --- |
| Chair disturbance | `chair_scene.py` | `chair_scene.xml` | `csv/0413_chair_falling.csv` |
| Sofa interaction | `sofa_scene.py` | `sofa_scene.xml`, `sofa.xml` | `csv/sofa_sitting.csv` |

From the repository root:

```sh
python unitree_robots/H2_URDF/task_3_furniture/chair_scene.py
python unitree_robots/H2_URDF/task_3_furniture/sofa_scene.py
```

Each runner has `EXPORT = False` by default. Enable it and select `SAVED_FILE_NAME` to collect a recording on viewer close. Paths are relative to the task, regardless of where Python is launched. The chair disturbance now targets the `chair` body; `seat` is a geometry rather than a body in the XML.

## Visualization and analysis

- `small_vis.ipynb` reads `csv/0413_chair_falling.csv`, plotting acceleration and angular velocity components/magnitudes and a selected interval. Change its input to inspect sofa recordings.
- `visualization.ipynb` compares `csv/sit_falling.csv` with `csv/0313data.csv`, using filtering, axis/unit conversion, and selected event markers. Document the provenance and coordinate conventions of the comparison recording before claiming physical validation.

The first notebook cell selects this task's working directory. Outputs were cleared during reorganization so plots must be regenerated from the current inputs. `qpos_sitting_sequence.csv` and other historical recordings are preserved, including generically named `test*.csv`; their provenance is not inferred from filenames. They are retained rather than discarded as potentially useful experimental evidence.

The existing scripts establish experimental poses and contact behavior. They do not yet establish a validated closed-loop sitting controller. Robot model and mesh assets are shared through `../shared/`.
