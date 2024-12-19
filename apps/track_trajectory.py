#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from affetto_nn_ctrl.control_utility import (
    RobotInitializer,
    Spline,
    create_controller,
    create_default_logger,
    release_pressure,
    resolve_joints_str,
)
from affetto_nn_ctrl.data_handling import (
    build_data_file_path,
    collect_files,
    copy_config,
    get_default_base_dir,
    get_default_counter,
    get_output_dir_path,
    prepare_data_dir_path,
)
from affetto_nn_ctrl.event_logging import event_logger, start_logging
from affetto_nn_ctrl.model_utility import (
    DefaultTrainedModelType,
    DelayStatesAllParams,
    DelayStatesParams,
    control_position_or_model,
    load_trained_model,
)
from affetto_nn_ctrl.plot_utility import extract_common_parts

if TYPE_CHECKING:
    from affctrllib import Logger

    from affetto_nn_ctrl import CONTROLLER_T

DEFAULT_N_REPEAT = 10


def track_motion_trajectory(
    controller: CONTROLLER_T,
    model: DefaultTrainedModelType | None,
    data_logger: Logger,
    reference: Spline,
    duration: float,
    data_file_path: Path,
    header_text: str,
    preview_steps: int,
    warmup_steps: int,
) -> None:
    qdes_func, dqdes_func = reference.get_qdes_func(), reference.get_dqdes_func()
    control_position_or_model(
        controller,
        model,
        qdes_func,
        dqdes_func,
        duration,
        preview_steps,
        data_logger,
        data_file_path,
        time_updater="accumulated",
        header_text=header_text,
        warmup_steps=warmup_steps,
    )
    data_logger.dump(quiet=True)


def save_paths(
    output_dir_path: Path,
    output_prefix: str,
    active_joints: list[int],
    model_file_path: str | None,
    reference_paths: list[Path],
    reference_output_paths: list[Path],
    motion_paths: dict[str, list[Path]],
    ext: str = ".toml",
) -> Path:
    if not ext.startswith("."):
        ext = f".{ext}"
    output = output_dir_path / f"{output_prefix}{ext}"

    joints_str = ", ".join(map(str, active_joints))
    text_lines = [
        "[model.performance]\n",
        f'model_path = "{model_file_path}"\n',
        f"active_joints = [ {joints_str} ]\n",
        "\n",
    ]
    for ref, ref_output in zip(reference_paths, reference_output_paths, strict=True):
        text_lines.extend(
            [
                f"[model.performance.{ref_output.stem}]\n",
                f'reference_path = "{ref!s}"\n',
                "\n",
            ],
        )
        for motion in motion_paths[ref_output.stem]:
            text_lines.extend(
                [
                    f"[[model.performance.{ref_output.stem}.errors]]\n",
                    f'motion_path = "{motion!s}"\n',
                    "\n",
                ],
            )
    with output.open("w") as f:
        f.writelines(text_lines)
    event_logger().info("Paths data saved: %s", output)
    return output


def run(  # noqa: PLR0915
    config: str,
    joints_str: list[str] | None,
    sfreq: float | None,
    cfreq: float | None,
    init_config: str | None,
    init_duration: float | None,
    init_duration_keep_steady: float | None,
    init_manner: str | None,
    q_init: list[float] | None,
    ca_init: list[float] | None,
    cb_init: list[float] | None,
    model_filepath: str | None,
    reference_files: list[str],
    glob_pattern: str,
    preview_steps: int,
    smoothness: float | None,
    given_duration: float | None,
    n_repeat: int,
    output_dir_path: Path,
    reference_prefix: str,
    output_prefix: str,
    *,
    overwrite: bool,
) -> None:
    # Create controller and data logger.
    comm, ctrl, state = create_controller(config, sfreq, cfreq)
    data_logger = create_default_logger(ctrl.dof)
    event_logger().debug("Loading config file: %s", config)
    event_logger().debug("Controller created: sfreq=%s cfreq=%s", state.freq, ctrl.freq)
    event_logger().debug("Default logger created: DOF=%s", ctrl.dof)

    # Initialize robot pose.
    initializer = RobotInitializer(
        ctrl.dof,
        config=init_config if init_config is not None else config,
        duration=init_duration,
        duration_keep_steady=init_duration_keep_steady,
        manner=init_manner,
        q_init=q_init,
        ca_init=ca_init,
        cb_init=cb_init,
    )
    initializer.get_back_home((comm, ctrl, state))
    q0 = state.q
    if initializer.get_manner() == "position":
        q0 = initializer.get_q_init()
    event_logger().debug("Initializer created: manner=%s", initializer.get_manner())
    event_logger().debug("Initial posture: %s", q0)

    # Resolve active joints.
    active_joints = resolve_joints_str(joints_str, dof=ctrl.dof)
    event_logger().debug("Resolved active joints: %s", active_joints)

    # Collect reference motion trajectories.
    event_logger().debug("Loading reference trajectories with following condition:")
    event_logger().debug("     Path list: %s", reference_files)
    event_logger().debug("  glob pattern: %s", glob_pattern)
    reference_paths = collect_files(reference_files, glob_pattern)
    event_logger().info("%s reference motion trajectories found", len(reference_paths))

    # Load trained model.
    model: DefaultTrainedModelType | None = None
    warmup_steps = 0
    if model_filepath is not None:
        model = load_trained_model(model_filepath)
        event_logger().info("Trained model is loaded: %s", model_filepath)
        event_logger().debug("Trained model: %s", model)
        if isinstance(model.adapter.params, DelayStatesParams | DelayStatesAllParams):
            warmup_steps = model.adapter.params.delay_step
            event_logger().debug("Warm-up step is set to: %d", warmup_steps)
    else:
        event_logger().info("No model is loaded, PID control is used.")
        event_logger().debug("Preview step is set to: %d", preview_steps)

    # Create reference file counter.
    n_reference = 0
    if not overwrite:
        n_reference = len(list(output_dir_path.glob(f"{reference_prefix}*")))
    cnt_reference = get_default_counter(n_reference)
    event_logger().debug("Reference counter initialized with %s", n_reference)

    reference_output_paths: list[Path] = []
    motion_paths: dict[str, list[Path]] = {}
    for i, reference_path in enumerate(reference_paths):
        reference_output_dir_path = build_data_file_path(output_dir_path, reference_prefix, cnt_reference, ext="")
        prepare_data_dir_path(reference_output_dir_path, make_latest_symlink=False)
        reference_output_paths.append(reference_output_dir_path)
        motion_paths[reference_output_dir_path.stem] = []

        # Create tracked motion trajectory counter.
        n = 0
        if not overwrite:
            n = len(list(reference_output_dir_path.glob(f"{output_prefix}*.csv")))
        cnt = get_default_counter(n)
        event_logger().debug("Tracked motion trajectory counter initialized with %s", n)

        # Load reference motion trajectory.
        reference = Spline(reference_path, active_joints, smoothness)
        event_logger().debug("Reference motion trajectory is loaded: %s", reference_path)
        duration = given_duration if given_duration is not None else reference.duration

        # Perform trajectory tracking.
        for j in range(n_repeat):
            if i != 0 or j != 0:
                # Initialize robot pose again.
                initializer.get_back_home((comm, ctrl, state))
            motion_file_path = build_data_file_path(
                reference_output_dir_path,
                prefix=output_prefix,
                iterator=cnt,
                ext=".csv",
            )
            header_text = f"[Ref:{i+1}/{len(reference_paths)}(Cnt:{j+1}/{n_repeat})] "
            header_text += "Performing trajectory tracking..."
            event_logger().debug(header_text)
            track_motion_trajectory(
                (comm, ctrl, state),
                model,
                data_logger,
                reference,
                duration,
                motion_file_path,
                header_text=header_text,
                preview_steps=preview_steps,
                warmup_steps=warmup_steps,
            )
            motion_paths[reference_output_dir_path.stem].append(motion_file_path)
            event_logger().debug("Motion file saved: %s", motion_file_path)

    # Release all joints.
    release_pressure((comm, ctrl, state))

    # Finish stuff.
    event_logger().debug("Tracking reference motion trajectory finished")
    comm.close_command_socket()
    state.join()

    # Save paths data
    save_paths(
        output_dir_path,
        output_prefix,
        active_joints,
        model_filepath,
        reference_paths,
        reference_output_paths,
        motion_paths,
    )


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trajectory tracking performance.",
    )
    default_base_dir = get_default_base_dir()
    # Configuration
    parser.add_argument(
        "-b",
        "--base-dir",
        default=str(default_base_dir),
        help="Base directory path for the current working project.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(default_base_dir / "config/affetto.toml"),
        help="Config file path for robot model.",
    )
    parser.add_argument(
        "-j",
        "--joints",
        nargs="*",
        help="Active joint index list allowing to move.",
    )
    parser.add_argument(
        "-F",
        "--sensor-freq",
        dest="sfreq",
        type=float,
        help="Sensor frequency.",
    )
    parser.add_argument(
        "-f",
        "--control-freq",
        dest="cfreq",
        type=float,
        help="Control frequency.",
    )
    parser.add_argument(
        "--init-config",
        help="Config file path for robot pose initializer.",
    )
    parser.add_argument(
        "--init-duration",
        type=float,
        help="Time duration for making the robot get back to home position.",
    )
    parser.add_argument(
        "--init-duration-keep-steady",
        type=float,
        help="Time duration for keep steady after making the robot get back to home position.",
    )
    parser.add_argument(
        "--init-manner",
        help="How to make the robot get back to the home position. Choose: [position, pressure]",
    )
    parser.add_argument(
        "--q-init",
        nargs="+",
        type=float,
        help="Joint angle list when making the robot get back to home position.",
    )
    parser.add_argument(
        "--ca-init",
        nargs="+",
        type=float,
        help="Valve command list for positive side when making the robot get back to home position.",
    )
    parser.add_argument(
        "--cb-init",
        nargs="+",
        type=float,
        help="Valve command list for negative side when making the robot get back to home position.",
    )
    # Input
    parser.add_argument(
        "model",
        nargs="?",
        help="Path to file in which trained model is encoded. If no model is provided, PID controller is used.",
    )
    parser.add_argument(
        "-r",
        "--reference-files",
        nargs="+",
        help="A directory path or paths to files storing reference motion trajectory.",
    )
    parser.add_argument(
        "-g",
        "--glob-pattern",
        default="*.csv",
        help="Glob pattern to filter motion files load as references.",
    )
    # Parameters
    parser.add_argument(
        "--preview-steps",
        type=int,
        default=0,
        help="Preview steps using for PID control.",
    )
    parser.add_argument(
        "-s",
        "--smoothness",
        type=float,
        help="Smoothing parameter to perform during spline interpolation.",
    )
    parser.add_argument(
        "-T",
        "--duration",
        type=float,
        help="Time duration to perform trajectory tracking.",
    )
    parser.add_argument(
        "-n",
        "--n-repeat",
        default=DEFAULT_N_REPEAT,
        type=int,
        help="Number of iterations to track each reference trajectory.",
    )
    # Output
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory where performed tracking data files are stored.",
    )
    parser.add_argument(
        "--reference-prefix",
        default="reference",
        help="Directory name prefix that will be added to reference motion trajectory.",
    )
    parser.add_argument(
        "--output-prefix",
        default="tracked_trajectory",
        help="Filename prefix that will be added to tracked motion files.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Boolean. If True, restart counter of generated data files and overwrite tracked motion files.",
    )
    parser.add_argument(
        "--label",
        default="track_performance",
        help="Label name of the current dataset.",
    )
    parser.add_argument(
        "--sublabel",
        help="Optional. Sublabel string for the current dataset.",
    )
    parser.add_argument(
        "--split-by-date",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Boolean. If True, split generated dataset by date.",
    )
    parser.add_argument(
        "--specify-date",
        help="Specify date string like '20240123T123456' or 'latest'. When the date string is specified, "
        "generated dataset will be stored in the specified date directory. When 'latest' is specified, "
        "find the latest directory.",
    )
    parser.add_argument(
        "--make-latest-symlink",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Boolean. If True, make a symbolic link to the latest.",
    )
    # Others
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable verbose console output. -v provides additional info. -vv provides debug output.",
    )
    return parser.parse_args()


def main() -> None:
    import sys

    args = parse()

    # Prepare input/output
    if args.output is not None:
        output_dir = Path(args.output)
    else:
        if args.model is None:
            if len(args.reference_files) == 1 and Path(args.reference_files[0]).is_file():
                base_dir = str(Path(args.reference_files[0]).parent)
            else:
                base_dir = str(extract_common_parts(*args.reference_files))
        else:
            base_dir = str(Path(args.model).parent)
        output_dir = get_output_dir_path(
            base_dir,
            None,
            None,
            args.label,
            args.sublabel,
            args.specify_date,
            split_by_date=args.split_by_date,
        )
    start_logging(sys.argv, output_dir, __name__, args.verbose)
    event_logger().info("Output directory: %s", output_dir)
    prepare_data_dir_path(output_dir, make_latest_symlink=args.make_latest_symlink)
    copy_config(args.config, args.init_config, None, output_dir)
    event_logger().debug("Parsed arguments: %s", args)

    # Start mainloop
    run(
        # configuration
        args.config,
        args.joints,
        args.sfreq,
        args.cfreq,
        args.init_config,
        args.init_duration,
        args.init_duration_keep_steady,
        args.init_manner,
        args.q_init,
        args.ca_init,
        args.cb_init,
        # input
        args.model,
        args.reference_files,
        args.glob_pattern,
        # parameters
        args.preview_steps,
        args.smoothness,
        args.duration,
        args.n_repeat,
        # output
        output_dir,
        args.reference_prefix,
        args.output_prefix,
        # boolean arguments
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "Cnt cb cfreq csv ctrl dataset dir env init mlp noqa pid sfreq sublabel symlink usr vv"
# End:
