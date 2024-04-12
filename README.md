<a id="top"></a>
![PYRO logo](data/logo/logo-transparent.png)

[![Generic badge](https://img.shields.io/badge/License-BSD3-blue)](LICENSE)
[![Static Badge](https://img.shields.io/badge/Python-3.11%2B-red?logoColor=red)](https://www.python.org/downloads/release/python-3110/)
[![Tests](https://github.com/marovira/pyro-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/marovira/pyro-ml/actions/workflows/tests.yml)

## What is Pyro?

Pyro is a light-weight package for training ML networks built on top of PyTorch. It is
designed to abstract all of the "boiler-plate" code involved with training. Specifically,
it wraps the following common patterns:

* Creation of the dataloaders.
* Initialization of CUDA, PyTorch, and random number states.
* Initialization for distributed training.
* Training, validation, and testing loops.
* Saving and loading checkpoints.
* Exporting to ONNX.

It is important to note that Pyro is **not** a fully fledged training environment similar
to [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning). Instead, Pyro
is focused on providing a simple and straight-forward interface that abstracts most of the
common code patterns while retaining the ability to be easily overridden to suit the
individual needs of each training scheme.

## Main Features

Pyro offers the following functionality out of the box:

1. Resume training: Pyro has been built with the ability to resume training if it is
   paused. Specifically, Pyro will ensure that the behaviour of the trained model is
   *identical* to the one it would've had if it had been trained without pauses.
2. Automatic detection of multi-GPU environments for distributed training. In addition,
   Pyro also supports training using `torchrun` and will automatically handle the
   initialisation and clean up of the distributed state. It will also correctly set the
   devices and maps to ensure weights are mapped tot he correct location.
3. Registries for creation of arbitrary types. These include: networks, loss functions,
   optimizers, schedulers, etc.
4. Correct handling of logging when doing distributed training (even over multiple nodes).

## Documentation

Documentation coming soon!

## Contributing

There are three ways in which you can contribute to Pyro:

* If you find a bug, please open an issue. Similarly, if you have a question
  about how to use it, or if something is unclear, please post an issue so it
  can be addressed.
* If you have a fix for a bug, or a code enhancement, please open a pull
  request. Before you submit it though, make sure to abide by the rules written
  below.
* If you have a feature proposal, you can either open an issue or create a pull
  request. If you are submitting a pull request, it must abide by the rules
  written below. Note that any new features need to be approved by me.

If you are submitting a pull request, the guidelines are the following:

1. Ensure that your code follows the standards and formatting of Pyro. The coding
   standards and formatting are enforced through the Ruff Linter and Formatter. Any
   changes that do not abide by these rules will be rejected. It is your responsibility to
   ensure that both Ruff and Mypy linters pass.
2. Ensure that *all* unit tests are working prior to submitting the pull
   request. If you are adding a new feature that has been approved, it is your
   responsibility to provide the corresponding unit tests (if applicable).

## License

Pyro is published under the BSD-3 license and can be viewed [here](LICENSE).
