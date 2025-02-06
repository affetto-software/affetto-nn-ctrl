#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from affetto_nn_ctrl.data_handling import prepare_data_dir_path
from affetto_nn_ctrl.event_logging import FakeLogger, event_logger, get_logging_level_from_verbose_count, start_logging

if TYPE_CHECKING:
    from collections.abc import Generator


if sys.version_info < (3, 11):
    import tomli as tomllib  # type: ignore[reportMissingImports]
else:
    import tomllib


@dataclass
class ScoreData:
    adapter_selector: str
    regressor_selector: str
    scaler_selector: str
    dataset_tag: str
    score_tag: str
    steps: int
    score_data_path: Path
    model_path: Path
    score_mean: float
    score_std: float
    test_datasets: list[Path]
    plot_paths: list[Path]
    scores: list[float]


def pop_path_name(fullpath: Path) -> Generator[str]:
    path = fullpath
    while str(path) != fullpath.root:
        last = path.name
        yield last
        path = path.parent


def load_score_data(score_data_path: Path) -> ScoreData:
    path_iter = pop_path_name(score_data_path)
    _ = next(path_iter)  # discard the first element
    score_tag = next(path_iter)
    dataset_tag = next(path_iter)
    scaler_selector = next(path_iter)
    regressor_selector = next(path_iter)
    adapter_selector = next(path_iter)
    steps = int(adapter_selector.split(".")[1][4:])
    with score_data_path.open("rb") as f:
        data = tomllib.load(f)
    performance_data = data["model"]["performance"]
    model_path = Path(performance_data["model_path"])
    score_mean = performance_data["score"]["mean"]
    score_std = performance_data["score"]["std"]
    test_datasets = [Path(x["test_dataset"]) for x in performance_data["scores"]]
    plot_paths = [Path(x["plot_path"]) for x in performance_data["scores"]]
    scores = [x["score"] for x in performance_data["scores"]]
    loaded_score_data = ScoreData(
        adapter_selector,
        regressor_selector,
        scaler_selector,
        dataset_tag,
        score_tag,
        steps,
        score_data_path,
        model_path,
        score_mean,
        score_std,
        test_datasets,
        plot_paths,
        scores,
    )
    event_logger().info("Score data loaded: %s", score_data_path)
    event_logger().debug("Loaded score data: %s", loaded_score_data)
    return loaded_score_data


_ScalerMap: TypeAlias = dict[str, ScoreData]
_AdapterMap: TypeAlias = dict[str, _ScalerMap]
CollectedScoreData: TypeAlias = dict[str, _AdapterMap]


def collect_score_data(
    basedir_list: list[str],
    adapter_list: list[str],
    regressor_list: list[str],
    scaler_list: list[str],
    step: int,
    dataset_tag: str,
    score_tag: str,
    filename: str,
) -> CollectedScoreData:
    collected_score_data: CollectedScoreData = {}
    for regressor, adapter, scaler in product(regressor_list, adapter_list, scaler_list):
        if adapter == "without-adapter":
            _adapter = "delay-states-all"
            _step = 0
        else:
            _adapter = adapter
            _step = step
        score_data_file = f"{_adapter}.step{_step:02d}/{regressor}/{scaler}/{dataset_tag}/{score_tag}/{filename}"
        found = False
        for basedir in basedir_list:
            score_data_path = Path(basedir) / score_data_file
            if score_data_path.exists() and not found:
                score_data = load_score_data(score_data_path)
                collected_score_data.setdefault(regressor, {}).setdefault(adapter, {})[scaler] = score_data
                found = True
            elif found:
                msg = f"Duplicate score data found: {score_data_path}"
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
        if not found:
            msg = f"No score data found: {score_data_file} in {basedir_list}"
            raise RuntimeError(msg)
    return collected_score_data


def generate_preamble() -> str:
    return r"""\documentclass{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{pdflscape}
\newcommand\mr[1]{\multicolumn{2}{r}{#1}} % handy shortcut macro
\begin{document}
"""


def generate_end() -> str:
    return r"""
\end{document}
"""


adapter_names = {
    "delay-states": "States delay",
    "delay-states-all": "Recursive states delay",
    "preview-ref": "Reference preview",
    "without-adapter": "W/o delay embedding",
}
scaler_names = {
    "none": "W/o scaler",
    "minmax": "MinMax",
    "maxabs": "MaxAbs",
    "std": "Std",
    "robust": "Robust",
}
regressor_names = {
    "linear.default": "Linear",
    "ridge.default": "Ridge",
    "mlp.default-iter800": "MLP(#100)/ReLU",
    "mlp.layer200-iter800": "MLP(#200)/ReLU",
    "mlp.layer100-100-iter800": "MLP(#100-#100)/ReLU",
    "mlp.layer100-100-iter800-identity": "MLP(#100-#100)/Identity",
    "mlp.layer100-100-iter800-logistic": "MLP(#100-#100)/Logistic",
    "mlp.layer100-100-iter800-logistic-lbfgs": "MLP(#100-#100)/Logistic/L-FBFGS",
    "mlp.layer100-100-iter800-logistic-sgd": "MLP(#100-#100)/Logistic/SGD",
    "mlp.layer100-100-iter800-tanh": "MLP(#100-#100)/tanh",
    "mlp.layer100-100-iter800-tanh-lbfgs": "MLP(#100-#100)/tanh/L-FBFGS",
    "mlp.layer100-100-iter800-tanh-sgd": "MLP(#100-#100)/tanh/SGD",
}

FontSize: TypeAlias = Literal[
    "tiny",
    "scriptsize",
    "footnotesize",
    "small",
    "normalsize",
    "large",
    "Large",
    "LARGE",
    "huge",
    "Huge",
]


def generate_table_wide(  # noqa: PLR0912,PLR0915,C901
    collected_score_data: CollectedScoreData,
    adapter_list: list[str],
    regressor_list: list[str],
    scaler_list: list[str],
    *,
    step: int,
    caption: str | None,
    label: str | None,
    font_size: FontSize,
    precision: int,
) -> str:
    n_adapter = len(adapter_list)
    n_scaler = len(scaler_list)
    n_column = n_adapter * n_scaler

    if caption is None:
        caption = r"Comparison of $R^{2}$ scores across regressor models. Delay/Preview step: " + str(step) + "."
    if label is None:
        label = "tab:r2-score-comparison-across-regressor"

    lines: list[str] = []
    lines.extend(
        [
            r"\begin{landscape}",
            r"\begin{table}",
            r"\setlength\tabcolsep{1pt}",
            f"\\{font_size}",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l *{" + str(n_column) + r"}{c} }",
            r"\toprule",
        ],
    )
    adapter_headers = [" "]
    adapter_headers.extend(
        [
            r"& \multicolumn{" + str(n_scaler) + r"}{c}{" + adapter_names.get(adapter, adapter) + r"}"
            for adapter in adapter_list
        ],
    )
    adapter_headers.append(r"\\")
    lines.append(" ".join(adapter_headers))

    midrule = [
        r"\cmidrule(lr){" + str(2 + i * n_scaler) + r"-" + str(1 + (i + 1) * n_scaler) + r"}"
        for i, _ in enumerate(adapter_list)
    ]
    lines.append(" ".join(midrule))

    scaler_headers = ["Regressor Models"]
    for _ in adapter_list:
        scaler_headers.extend([f"& {scaler_names.get(scaler, scaler)}" for scaler in scaler_list])
    scaler_headers.append(r"\\")
    lines.append(" ".join(scaler_headers))
    lines.append(r"\midrule")

    for regressor in regressor_list:
        scores: list[float] = []
        line: list[str] = []
        regressor_name = regressor_names.get(regressor, regressor)
        # mean values
        line.append(regressor_name)
        for adapter in adapter_list:
            _line: list[str] = []
            for scaler in scaler_list:
                score_data = collected_score_data.get(regressor, {}).get(adapter, {}).get(scaler, None)
                if score_data is not None:
                    scores.append(score_data.score_mean)
                    _line.append(f"& {score_data.score_mean:.{precision}f}")
                else:
                    scores.append(1e-32)  # very small value
                    _line.append("& ---")
            line.extend(_line)
        argmax = np.argmax(scores) + 1
        line[argmax] = r"& \textbf{" + line[argmax][2:] + r"}"
        line.append(r"\\")
        lines.append(" ".join(line))
        # deviation values
        line = [" " * len(regressor_name)]
        for adapter in adapter_list:
            _line = []
            for scaler in scaler_list:
                score_data = collected_score_data.get(regressor, {}).get(adapter, {}).get(scaler, None)
                if score_data is not None:
                    _line.append(f"& ({score_data.score_std:.{precision}f})")
                else:
                    _line.append("& ")
            line.extend(_line)
        if regressor != regressor_list[-1]:
            line.append(r"\\[3pt]")
        else:
            line.append(r"\\")
        lines.append(" ".join(line))

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular*}",
            r"\end{table}",
            r"\end{landscape}",
        ],
    )
    return "\n".join(lines)


def generate_latex(
    basedir_list: list[str],
    adapter_list: list[str],
    regressor_list: list[str],
    scaler_list: list[str],
    step: int,
    dataset_tag: str,
    score_tag: str,
    filename: str,
    *,
    caption: str | None,
    label: str | None,
    font_size: FontSize,
    precision: int,
) -> str:
    collected_score_data = collect_score_data(
        basedir_list,
        adapter_list,
        regressor_list,
        scaler_list,
        step,
        dataset_tag,
        score_tag,
        filename,
    )
    table = generate_table_wide(
        collected_score_data,
        adapter_list,
        regressor_list,
        scaler_list,
        step=step,
        caption=caption,
        label=label,
        font_size=font_size,
        precision=precision,
    )
    return f"""\
{generate_preamble()}
{table}
{generate_end()}
"""


def write(
    basedir_list: list[str],
    adapter_list: list[str],
    regressor_list: list[str],
    scaler_list: list[str],
    step: int,
    dataset_tag: str,
    score_tag: str,
    filename: str,
    *,
    caption: str | None,
    label: str | None,
    font_size: FontSize,
    precision: int,
    output: Path,
) -> None:
    latex_code = generate_latex(
        basedir_list,
        adapter_list,
        regressor_list,
        scaler_list,
        step,
        dataset_tag,
        score_tag,
        filename,
        caption=caption,
        label=label,
        font_size=font_size,
        precision=precision,
    )
    with output.open("w"):
        output.write_text(latex_code)


def print_instruction(output: Path) -> None:
    sys.stderr.write(f"Generated LaTeX code is saved in '{output}'.\n")
    sys.stderr.write("To generate PDF file, run the following command:\n")
    sys.stderr.write("--\n")
    sys.stderr.write(f"pdflatex {output}\n")


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LaTeX table to compare scores across regressor models")
    parser.add_argument("basedir", nargs="+", help="List of paths to directories containing scores data.")
    parser.add_argument("-i", "--step", type=int, default=9, help="Delay/Preview step to show in table.")
    parser.add_argument("-a", "--adapter", nargs="+", help="Data adapter selector.")
    parser.add_argument("-s", "--scaler", nargs="+", help="Scaler selector.")
    parser.add_argument("-r", "--regressor", nargs="+", help="Regressor selector.")
    parser.add_argument("-d", "--dataset-tag", required=True, help="Dataset tag.")
    parser.add_argument("--score-tag", default="scores_000", help="Score data tag. (default: scores_000)")
    parser.add_argument("--score-filename", default="scores.toml", help="Scores filename. (default: scores.toml)")
    parser.add_argument("--caption", help="caption")
    parser.add_argument("--label", help="label")
    parser.add_argument(
        "--font-size",
        choices=[
            "tiny",
            "scriptsize",
            "footnotesize",
            "small",
            "normalsize",
            "large",
            "Large",
            "LARGE",
            "huge",
            "Huge",
        ],
        default="normalsize",
        help="font size in table",
    )
    parser.add_argument("--precision", type=int, default=3, help="precision for floating value")
    parser.add_argument("-o", "--output", help="Path to filename that generated LaTeX code is saved.")
    parser.add_argument(
        "--show-instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to show instructions to compile generated LaTeX file.",
    )
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
    if args.verbose > 0 and isinstance(event_logger(), FakeLogger):
        event_logger().setLevel(get_logging_level_from_verbose_count(args.verbose))

    output: Path
    if args.output is not None:
        output = Path(args.output)
    else:
        output = Path(args.basedir[0]) / f"score_comparison_table_step_{args.step}.tex"
    start_logging(sys.argv, output.parent, __name__, args.verbose)
    event_logger().info("Output file: %s", output)
    prepare_data_dir_path(output.parent, make_latest_symlink=False)
    event_logger().debug("Parsed arguments: %s", args)

    write(
        args.basedir,
        args.adapter,
        args.regressor,
        args.scaler,
        args.step,
        args.dataset_tag,
        args.score_tag,
        args.score_filename,
        caption=args.caption,
        label=args.label,
        font_size=args.font_size,
        precision=args.precision,
        output=output,
    )
    if args.show_instruction:
        print_instruction(output)


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "ReLU basedir booktabs bottomrule cmidrule dataset documentclass env extracolsep footnotesize lbfgs linewidth lr maxabs midrule minmax mlp mr multicolumn newcommand noqa normalsize pdflatex pdflscape rb regressor scaler scriptsize setlength sgd tabcolsep tanh textbf toprule usepackage usr vv" # noqa: E501
# End:
