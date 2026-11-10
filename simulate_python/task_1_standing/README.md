# Task 1: Standing fall detection

Question: can acceleration, angular velocity, and ToF measurements identify a developing fall?

## Trace the workflow

| Stage | Implementation | Input / output |
| --- | --- | --- |
| Inspect simulation | `h1_2_sim.py` | H1-2 scene configured in `config.py` |
| Collect episodes | `h1_2_sim_headless.py` | Random disturbances -> `csv/<episode>.csv` |
| Inspect sensors | `visualization.ipynb` | `csv/1.csv` -> acceleration, gyro, ToF plots |
| Train LSTM | `training.ipynb` | `csv/0.csv` through `1999.csv` -> `0210.keras` |
| Interactive inference | `h1_2_sim_keras.py` | Saved model and simulated sensor windows |
| Batch inference | `h1_2_sim_headless_keras.py` | `0210.keras` -> `time_data.csv` |
| Offline check | `keras_test.py` | `csv/1237.csv` and saved model |
| Analyze timing | `time_data_processing.ipynb` | `time_data.csv` -> timing plots |
| Export model | `quant_convert_tflite.ipynb` | Keras -> TFLite and C header |

From the repository root:

```sh
python simulate_python/task_1_standing/h1_2_sim.py
python simulate_python/task_1_standing/h1_2_sim_headless.py
python simulate_python/task_1_standing/h1_2_sim_keras.py
python simulate_python/task_1_standing/h1_2_sim_headless_keras.py
```

Open notebooks in the order above. Their first cell sets the working directory to this task. Scripts resolve the robot scene, datasets, and saved model from their own location. `config.py` contains DDS interface settings; choose an interface available on your machine. The shared SDK bridge remains next to the standing scripts.

## Data and evaluation notes

The original `csv/`, `0227csvs/`, `dedicated_csvs/`, trained model, exported model, and timing data are preserved. The latter two dataset directories are supplementary historical recordings; their provenance should be documented before using them in evaluation. Collection and training can overwrite existing data/model files; copy recordings before rerunning those stages.

The current notebook uses time > 0.2 seconds as its label rule and tests on episode 1237, which is also in its training input. These are existing experimental limitations, not a held-out benchmark. Split by episode before window generation and report independent test results before making accuracy claims. Separate detection time, lead time before impact, and inference runtime when presenting timing results. Model export alone does not establish embedded deployment performance.
