.. -*- mode: rst -*-

.. image:: https://raw.githubusercontent.com/marovira/helios-ml/master/data/logo/logo-transparent.png
   :target: https://github.com/marovira/helios-ml

|Test| |CodeFactor| |Ruff| |PythonVersion| |PyPi| |License|

.. |Test| image:: https://github.com/marovira/helios-ml/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/marovira/helios-ml/actions/workflows/tests.yml

.. |CodeFactor| image:: https://www.codefactor.io/repository/github/marovira/helios-ml/badge
   :target: https://www.codefactor.io/repository/github/marovira/helios-ml

.. |Ruff| image:: https://img.shields.io/badge/code%20style-ruff-red
   :target: https://github.com/astral-sh/ruff

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/helios-ml.svg
   :target: https://pypi.org/project/helios-ml/

.. |PyPi| image:: https://img.shields.io/pypi/v/helios-ml.svg
   :target: https://pypi.org/project/helios-ml/

.. |License| image:: https://img.shields.io/pypi/l/helios-ml.svg
   :target: https://opensource.org/license/bsd-3-clause

.. |PythonMinVersion| replace:: 3.11
.. |TQDMMinVersion| replace:: 4.66.2
.. |OpenCVMinVersion| replace:: 4.9.0.80
.. |TensorboardMinVersion| replace:: 2.16.2
.. |TorchMinVersion| replace:: 2.2.1
.. |TorchvisionMinVersion| replace:: 0.17.1
.. |ONNXMinVersion| replace:: 1.16.0
.. |ORTMinVersion| replace:: 1.17.1
.. |PLTMinVersion| replace:: 3.8.4

What is Helios?
---------------

Named after Greek god of the sun, Helios is a light-weight package for training ML
networks built on top of PyTorch. It is designed to abstract all of the "boiler-plate"
code involved with training. Specifically, it wraps the following common patterns:

- Creation of the dataloaders.
- Initialization of CUDA, PyTorch, and random number states.
- Initialization for distributed training.
- Training, validation, and testing loops.
- Saving and loading checkpoints.
- Exporting to ONNX.

It is important to note that Helios is **not** a fully fledged training environment similar
to `Pytorch Lightning <https://github.com/Lightning-AI/pytorch-lightning>`__. Instead,
Helios is focused on providing a simple and straight-forward interface that abstracts most
of the common code patterns while retaining the ability to be easily overridden to suit
the individual needs of each training scheme.

Main Features
~~~~~~~~~~~~~

Helios offers the following functionality out of the box:

1. Resume training: Helios has been built with the ability to resume training if it is
   paused. Specifically, Helios will ensure that the behaviour of the trained model is
   *identical* to the one it would've had if it had been trained without pauses.
2. Automatic detection of multi-GPU environments for distributed training. In addition,
   Helios also supports training using ``torchrun`` and will automatically handle the
   initialisation and clean up of the distributed state. It will also correctly set the
   devices and maps to ensure weights are mapped tot he correct location.
3. Registries for creation of arbitrary types. These include: networks, loss functions,
   optimizers, schedulers, etc.
4. Correct handling of logging when doing distributed training (even over multiple nodes).

Installation
------------

Dependencies
~~~~~~~~~~~~

Helios requires:

- Python (>= |PythonMinVersion|)
- TQDM (>= |TQDMMinVersion|)
- OpenCV (>= |OpenCVMinVersion|)
- Tensorboard (>= |TensorboardMinVersion|)
- PyTorch (>= |TorchMinVersion|)
- Torchvision (>= |TorchvisionMinVersion|)
- ONNX (>= |ONNXMinVersion|)
- ONNXRuntime (>= |ORTMinVersion|)
- Matplotlib (>= |PLTMinVersion|)

User Installation
~~~~~~~~~~~~~~~~~

You can install Helios using ``pip``::

    pip install -U helios-ml

If you require a specific version of CUDA, you can install with::

    pip install -U helios-ml --extra-index-url https://download.pytorch.org/whl/cu<version>

Documentation
-------------

Documentation coming soon!

Contributing
------------

There are three ways in which you can contribute to Helios:

- If you find a bug, please open an issue. Similarly, if you have a question
  about how to use it, or if something is unclear, please post an issue so it
  can be addressed.
- If you have a fix for a bug, or a code enhancement, please open a pull
  request. Before you submit it though, make sure to abide by the rules written
  below.
- If you have a feature proposal, you can either open an issue or create a pull
  request. If you are submitting a pull request, it must abide by the rules
  written below. Note that any new features need to be approved by me.

If you are submitting a pull request, the guidelines are the following:

1. Ensure that your code follows the standards and formatting of Helios. The coding
   standards and formatting are enforced through the Ruff Linter and Formatter. Any
   changes that do not abide by these rules will be rejected. It is your responsibility to
   ensure that both Ruff and Mypy linters pass.
2. Ensure that *all* unit tests are working prior to submitting the pull
   request. If you are adding a new feature that has been approved, it is your
   responsibility to provide the corresponding unit tests (if applicable).

License
-------

Helios is published under the BSD-3 license and can be viewed
`here <https://raw.githubusercontent.com/marovira/helios-ml/master/LICENSE>`__.
