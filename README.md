# affetto-nn-ctrl

A data-driven controller using neural networks for Affetto.

## Description

This project aims to develop a data-driven controller using neural
networks for Affetto. This repository contains programs to collect
sensory / actuation data when moving Affetto at random, record a
trajectory of joint angles for kinesthetic teaching, train a
multi-layered perceptron model, evaluate a tracking performance among
controllers based on trained neural networks and conventional PID
controller.

## Getting Started

### Dependencies

The software is implemented in Python and developed under management
of [Rye](https://rye.astral.sh/). To install `rye` run the following
command:

``` shell
curl -sSf https://rye.astral.sh/get | bash
```

### Installing

To setup a virtual environment to execute programs, just clone this
repository and install dependencies.

``` shell
git clone https://github.com/hrshtst/affetto-nn-ctrl.git
cd affetto-nn-ctrl
rye sync
```

The `rye run` command provides execution of programs in a virtual
environment, for example:

``` shell
rye run python ./apps/collect_data.py
```

## Application Usage

### Collecting data

### Recording a reference trajectory

### Training a model

### Evaluating tracking performances


## License

This project is licensed under the MIT License.
