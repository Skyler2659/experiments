"""Compatibility API for observation evidence; implementation lives with CodeWorld."""
from codeworld_workspace.observation_evidence import assess_grasp, load_capture, masked_cloud

# Existing external callers may still use these names.
_capture = load_capture
_cloud = masked_cloud
