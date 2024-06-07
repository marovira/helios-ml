.. Helios documentation master file, created by
   sphinx-quickstart on Thu Jun  6 15:38:41 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Helios's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

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
-------------

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
