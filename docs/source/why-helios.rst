Why Helios?
###########

Helios is built around two principles: explicitness and simplicity. It removes training
boilerplate without hiding what is happening. Every step in the training loop is visible
and overridable, which makes the code easy to follow and debug.

This page explains how Helios compares to other frameworks, describes its registry
system in detail, and helps you decide whether it is the right tool for your project.

.. _comparison:

Comparison with Other Frameworks
=================================

Compared to larger frameworks, Helios prioritises explicitness and simplicity over
automation. Rather than inferring behaviour from annotations or hooks, Helios requires
you to state your intent explicitly in code you own and can read.

The table below contrasts Helios with PyTorch Lightning and Ignite across the most
important areas for research and engineering work:

.. list-table::
   :header-rows: 1
   :widths: 30 23 23 24

   * - Feature
     - Helios
     - Lightning
     - Ignite
   * - Mixed precision
     - Helper functions
     - Automatic
     - Manual
   * - Reproducible resume
     - Built-in
     - Partial
     - Manual
   * - Distributed training
     - ``torchrun``
     - ``torchrun``
     - Manual
   * - Boilerplate style
     - Explicit
     - Magic hooks
     - Event-based
   * - Training unit
     - First-class (``EPOCH``/``ITERATION``)
     - Epoch-based
     - Epoch-based
   * - Gradient accumulation
     - Iteration-aware
     - Epoch-based
     - Manual
   * - Registry system
     - Built-in
     - None
     - None
   * - Learning curve
     - Low
     - Medium
     - High
   * - Debuggability
     - High
     - Medium
     - High

**Mixed precision.** Helios exposes
:py:func:`~helios.model.model.Model.create_scaler`,
:py:func:`~helios.model.model.Model.autocast`, and
:py:func:`~helios.model.model.Model.clip_gradients` as explicit helper functions. You
opt in deliberately and interact with the scaler directly, so the behaviour is always
clear. Lightning applies mixed precision automatically based on a trainer flag while
Ignite leaves it entirely to the user.

**Reproducible resume.** When resuming from a checkpoint, Helios guarantees that three
things are restored to exactly the state they were in when the checkpoint was saved: the
training state (internal and user-defined), the RNG state, and the sequence of batches.
The last point is the key differentiator: Helios uses resumable samplers by default, so
the dataloader picks up from exactly the same position in the dataset. Lightning saves
core training state and RNG state but does not provide the batch sequence guarantee by
default. Ignite provides no built-in resumption support.

**Distributed training.** Both Helios and Lightning support ``torchrun`` with automatic
device detection. Ignite requires manual process group setup.

**Boilerplate style.** Lightning relies on magic hook names and decorator-based
injection; Ignite uses an event system. Helios uses explicit function overrides with
clear call-site visibility.

**Training unit.** In Helios, the training unit is a first-class concept.
:py:attr:`~helios.trainer.TrainingUnit.EPOCH` and
:py:attr:`~helios.trainer.TrainingUnit.ITERATION` are distinct modes that govern the
entire training loop, including checkpoint frequency, stopping conditions, and gradient
accumulation behaviour. Both Lightning and Ignite are fundamentally epoch-based;
Lightning exposes ``max_steps`` as a secondary knob, but iteration-based training in
either framework requires working against the grain of the default design.

**Gradient accumulation.** Helios handles gradient accumulation through the
:py:class:`~helios.trainer.TrainingState` rather than a dedicated parameter. In epoch
mode, the trainer does not intervene: the iteration count is unchanged and the model
decides when to call ``backward()`` by inspecting
:py:attr:`~helios.trainer.TrainingState.current_iteration`. In iteration mode, the
trainer automatically scales the total iteration count by the accumulation factor, so
requesting N iterations with accumulation M truly means N backward passes at the target
batch size. The distinction between
:py:attr:`~helios.trainer.TrainingState.current_iteration` (effective, complete
iterations) and :py:attr:`~helios.trainer.TrainingState.global_iteration` (raw forward
passes) makes the accounting explicit. See :ref:`Gradient Accumulation <gradient-accumulation>`
in the quick reference for the full details.

.. _registry:

The Registry System
====================

Helios provides typed global registries that map string names to types. Registries exist
for all major components:

* :py:data:`~helios.model.MODEL_REGISTRY`
* :py:data:`~helios.data.DATASET_REGISTRY`
* :py:data:`~helios.optim.OPTIMIZER_REGISTRY`
* :py:data:`~helios.scheduler.SCHEDULER_REGISTRY`
* :py:data:`~helios.losses.LOSS_REGISTRY`
* :py:data:`~helios.plugins.PLUGIN_REGISTRY`
* :py:data:`~helios.data.COLLATE_FN_REGISTRY`
* :py:data:`~helios.data.TRANSFORM_REGISTRY`

Any component can be registered with the ``@REGISTRY.register`` decorator and
instantiated by name using the corresponding factory function:

.. code-block:: python

   import helios.model as hlm

   @hlm.MODEL_REGISTRY.register
   class MyModel(hlm.Model):
       ...

   model = hlm.create_model("MyModel", save_name="my_model")

This enables config-file-driven experiments where the model, dataset, optimiser, and
scheduler are all selected by string name, with no changes to training code. Swapping
any component is a one-line change in the configuration.

If your source tree spans multiple packages, use
:py:func:`~helios.core.registry.update_all_registries` to scan and auto-register all
decorated classes at startup:

.. code-block:: python

   import helios.core as hlc

   hlc.update_all_registries("my_package")

.. _when_to_use:

When to Choose Helios
======================

Helios is a good fit if:

* You need exact reproducibility. Pausing and resuming training must produce results
  identical to an uninterrupted run.
* You run config-file-driven experiments and want to swap components by string name.
* You want to be able to read and follow every line of the training loop.
* Your training logic is non-standard (for example, a GAN, a multi-phase curriculum, or
  a custom distributed setup) and does not fit neatly into a higher-level abstraction.

When Lightning Might Be a Better Fit
======================================

Lightning may be more appropriate if:

* You want a batteries-included experience and would prefer not to write any training
  boilerplate at all.
* You are working within an ecosystem that already depends on Lightning, such as a team
  codebase or a set of third-party integrations built around it.
