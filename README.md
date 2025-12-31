# affetto-nn-ctrl

A data-driven controller using neural networks for Affetto.

## Description

This project develops a data-driven controller for Affetto that compensates for input delays using neural networks. The repository includes tools to:

* Collect sensory and actuation data during random movements of Affetto.
* Record joint angle trajectories through kinesthetic teaching.
* Train a multi-layer perceptron (MLP) model that compensates for specific delays.
* Evaluate and compare the tracking performance of neural network-based controllers against a conventional PID controller.

## Getting Started

### Dependencies

The software is written in Python and managed with [uv](https://docs.astral.sh/uv/). To install `uv`, run:

``` shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

Clone the repository and install dependencies in a virtual environment:

``` shell
git clone https://github.com/affetto-software/affetto-nn-ctrl.git
cd affetto-nn-ctrl
uv sync
```

You can run programs in the virtual environment using:

``` shell
uv run python apps/collect_data.py -h
```

### Configuration

If you use a shared external storage location for data, you can specify a default output directory by creating a file `apps/base_dir`:

``` shell
echo "/mnt/data-storage/somewhere" > apps/base_dir
```

If `base_dir` is not specified and the `-b/--base-dir` option is not provided, the output directory defaults to `./data`.

### Running Tests

To execute the test suite, run:

``` shell
uv run pytest
```

## Application Usage

### Data Collection

To collect random motion data, use `apps/collect_data.py`. For example, to collect data for the left elbow (joint ID: 5) for 60 seconds over 100 trials:

Use `apps/collect_data.py` to collect random motion data. The example below collects data from the left elbow (joint ID: 5) for 60 seconds, repeated 100 times. It uses a step-wise reference update pattern (from 0.1 to 1.0 seconds) and a position range of 20--40%, in sync mode:

``` shell
uv run python apps/collect_data.py -v \
  --init-config "apps/config/init_left_elbow.toml" \
  --config "apps/config/affetto.toml" \
  --joints 5 -T 60 -s 12345 -t 0.1 1.0 -q 20 40 \
  -p step --no-async-mode -n 100 \
  --label "left_elbow" --sublabel "step/sync/fast" --no-overwrite
```

For additional options, refer to the help message.

### Model Training

Use `apps/train_model.py` to train a model using collected data. The example below trains an MLP using reference preview preprocessing and standard deviation scaling. It selects 25% of the dataset at random. A fixed seed, e.g., 42, ensures reproducibility:

``` shell
label="left_elbow"; cont=step; sync=sync; scale=fast
a=preview-ref.default; s=std.default; r=mlp.default
uv run python apps/train_model.py -v \
  /mnt/data-storage/somewhere/affetto_nn_ctrl/dataset/left_elbow/20241203T144115/${cont}/${sync}/${scale} \
  -j 5 --train-size 0.25 --seed 42 \
  -m apps/config/model.toml \
  -a ${a} -s ${s} -r ${r} \
  --label "${label}" --sublabel "${a}/${r}/${s}/${cont}_${sync}_${scale}"
```

See the help message and `model.toml` for more details.

### Prediction Evaluation

Use `apps/calculate_score.py` to compute prediction performance with $R^2$ scores on unseen data. This example samples 10% of the files from each of the `slow`, `middle`, and `fast` motion datasets:

``` shell
label="left_elbow"; cont=step; sync=sync; scale=fast
a=preview-ref.default; s=std.default; r=mlp.default
uv run python apps/calculate_score.py -v \
  /mnt/data-storage/somewhere/affetto_nn_ctrl/trained_model/left_elbow/latest/${a}/${r}/${s}/${cont}_${sync}_${scale}/trained_model.joblib \
  -d /mnt/data-storage/somewhere/affetto_nn_ctrl/dataset/left_elbow/20241203T144115/${cont}/${sync}/{slow,middle,fast} \
  --test-size 0.1 --split-in-each-directory --seed 42 \
  -e png pdf
```

This script saves scores in `scores.toml` and time-series prediction plots are generated.

### Recording Reference Trajectories

Use `apps/record_trajectory.py` to record joint position trajectories. The following example records the left elbow joint for 30 seconds:

``` shell
uv run python apps/record_trajectory.py -v \
  --init-config "apps/config/init_left_elbow.toml" \
  --joints 5 -T 30 --label left_elbow
```

### Tracking Performance Evaluation

Use `apps/track_trajectory.py` to evaluate the tracking performance of a trained model. The following command replays a specific reference trajectory, repeated 10 times:

``` shell
label="left_elbow"; cont=step; sync=sync; scale=fast
a=preview-ref.default; s=std.default; r=mlp.default
uv run python apps/track_trajectory.py -v \
  /mnt/data-storage/somewhere/affetto_nn_ctrl/trained_model/left_elbow/latest/${a}/${r}/${s}/${cont}_${sync}_${scale}/trained_model.joblib \
  --init-config "apps/config/init_left_elbow.toml" \
  --joints 5 \
  -r /mnt/data-storage/somewhere/affetto_nn_ctrl/reference_trajectory/left_elbow/20241219T111141/reference_trajectory_000.csv \
  -n 10
```

To compare with the conventional PID controller, omit the model path. The following command tracks the same reference trajectory through PID control:

``` shell
uv run python apps/track_trajectory.py -v \
  --init-config "apps/config/init_left_elbow.toml" \
  --joints 5 \
  -r /mnt/data-storage/somewhere/affetto_nn_ctrl/reference_trajectory/left_elbow/20241219T111141/reference_trajectory_000.csv \
  -n 10
```

Results are saved in `tracked_trajectory.toml`.

### Calculating RMSE

Use `apps/calculate_rmse.py` to compute the root mean squared error:
To compute the Root Mean Squared Error (RMSE) between actual and reference trajectories, use `apps/calculate_rmse.py`:

``` shell
label="left_elbow"; cont=step; sync=sync; scale=fast
a=preview-ref.default; s=std.default; r=mlp.default
uv run python apps/calculate_rmse.py -v \
  /mnt/data-storage/somewhere/affetto_nn_ctrl/trained_model/left_elbow/latest/${a}/${r}/${s}/${cont}_${sync}_${scale}/track_performance/latest/tracked_trajectory.toml \
  --fill --fill-err-type std --fill-alpha 0.2 \
  -e png pdf
```

RMSE values are added to `tracked_trajectory.toml`, and time-series plots are saved in the output directory.

## Demonstration Instruction

### Kinesthetic Teaching Demonstration

Performing a simple kinesthetic teaching demonstration is attractive
for visitors. This demonstration consists of the following three
steps:
- Teach a motion that the robot reproduce.
- Track the recorded motion with a trained model.
- (Optional) Track the same motion with PID control for comparison.

The following instruction uses 4 joints in the left arm of Affetto.

#### Teaching

Use `apps/record_trajectory.py` to record joint position trajectories.
The following example records the left arm joints for 10 seconds:

``` shell
uv run python apps/record_trajectory.py -v --init-config "apps/config/init_left_arm.toml" --joints 2-5 -T 10 --base-dir "." --output "demo/$(date '+%Y%m%dT%H%M%S')"
```

#### Playback using trained model

Use `apps/track_trajectory.py` to perform the trajectory tracking using a trained model. The following command replays a specific reference trajectory:

``` shell
model="/mnt/storage1/data/affetto_nn_ctrl/trained_model/left_arm/20250207T001027/preview-ref.step09/mlp.layer200-iter1000-logistic/std/trapez_async_all/trained_model.joblib"
uv run python apps/track_trajectory.py -v "$model" --init-config "apps/config/init_left_arm.toml" --joints 2-5 -r "./demo/date_time_when_recorded/reference_trajectory_000.csv" -n 1 --base-dir "." --output "demo/$(date '+%Y%m%dT%H%M%S')" --output-prefix "tracked_trajectory_mlp"
```
#### (Optional) Plot tracked trajectories

Use `apps/drawer/compare_tracked_trajectories.py` to visualize the tracked trajectories. The following commands show plots:

``` shell
uv run python apps/calculate_rmse.py -v "./demo/date_time_when_played/tracked_trajectory_mlp.toml" -j 2-5 --output-prefix "tracked_trajectory_mlp"
uv run python apps/drawer/compare_tracked_trajectories.py -v "./demo/date_time_when_played/tracked_trajectory_mlp.toml" -j 2-5 -e png --show-screen
```

#### (Optional) Playback using PID control

Depending on reactions from visitors, tracking the same recorded
trajectory with the conventional PID controller might be convincing. The following command tracks the same reference trajectory through PID control:

``` shell
uv run python apps/track_trajectory.py -v --init-config "apps/config/init_left_arm.toml" --joints 2-5 -r "./demo/date_time_when_recorded/reference_trajectory_000.csv" -n 1 --base-dir "." --output "demo/$(date '+%Y%m%dT%H%M%S')" --output-prefix "tracked_trajectory_pid"
```

## License

This project is licensed under the MIT License.
