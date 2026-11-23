"""Experiments on tool-using LLM auditor agents: virtual black-box barriers and
computational Blackwell inversion.

Experiment 1 (`C_kXOR`) asks an agent whether a Boolean circuit hides a planted noisy
k-XOR backdoor; Experiment 2 (`C_inv`) asks it to recover a hidden 3-variable rule whose
white-box form contains a readable decoy branch that never fires. Both compare black-box
(oracle only) with white-box (oracle plus code or weights) access under the same budget.
"""

__version__ = "1.0.0"
