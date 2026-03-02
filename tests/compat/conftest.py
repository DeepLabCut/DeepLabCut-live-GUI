# tests/compat/conftest.py
import sys
import types

# Stub out torch imports to avoid ImportError when torch is not installed in DLCLive package.
# This allows testing of DLCLive API compatibility without requiring torch.
# Ideally imports should be guarded in the package itself, but this is a pragmatic solution for now.
# IMPORTANT NOTE: This should ideally be removed and replaced whenever possible.
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")
