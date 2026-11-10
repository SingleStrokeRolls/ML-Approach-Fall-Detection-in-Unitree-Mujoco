# Task 2: Directional falls onto cushions

Question: how do scripted arm and body responses change fall motion and sensor signatures?

## Trace the workflow

1. In `start_living_room.py`, choose `SCENARIO` from `SCENARIO_LIST` and set a distinct `FILE_NAME`.
2. The runner selects the backward, forward, or leftward `scene_with_cushion*.xml` from the scenario prefix.
3. With `SAVE = True`, closing the viewer exports IMU and ToF samples to this task's `csv/` directory.
4. Run `visualization.ipynb` and select the corresponding CSV. Its default is `0413_leftward_with_arms.csv`.

From the repository root:

```sh
python unitree_robots/H2_URDF/task_2_cushion/start_living_room.py
```

| Direction | Existing scenarios | Recorded examples |
| --- | --- | --- |
| Backward | `BACKWARD`, `BACKWARD_TWIST` | `0413_backward_twist.csv` |
| Forward | `FORWARD_FREE_FALL`, `FORWARD_WITH_ARMS` | Matching `0413_forward_*.csv` |
| Leftward | `LEFTWARD_FREE_FALL`, `LEFTWARD_WITH_ARMS` | Matching `0413_leftward_*.csv` |

`straight_falling.csv` and `test_with_cushioned.csv` are preserved historical recordings. Establish their setup before treating them as matched comparisons. The notebook plots acceleration and angular velocity components and magnitudes, including a selected time range. It was adapted from the existing H2 sensor notebook; stale cell outputs were cleared.

## Supporting material experiment

`material_validation/` preserves the original ball/cushion test, CSVs, and its own `visualization.ipynb`. Run `material_validation/test.py` to generate its recording. This is a supporting contact experiment within task 2.

## Interpretation

Some responses directly write joint positions, and disturbances/initial poses differ between existing scenarios. Describe these as scripted experiments. Match initial conditions before claiming a control strategy improves outcomes. Current CSVs contain sensor samples, not contact forces. Shared robot assets and CSV export utilities are in `../shared/`.
