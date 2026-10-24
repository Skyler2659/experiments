"""Resolve operator-configured integration hooks outside the generic robot layer."""

from importlib import import_module


def load_hook(reference):
    module, name = reference.split(":", 1)
    hook = getattr(import_module(module), name)
    if not callable(hook):
        raise TypeError(f"Integration hook is not callable: {reference}")
    return hook
