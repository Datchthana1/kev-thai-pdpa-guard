"""Windows shims for kev (written for POSIX): kev.train does `import resource` and calls getrusage(...).ru_maxrss (KiB)."""
import os
import sys
import types


def install_resource_shim():
    try:
        import resource  # noqa: F401  (POSIX)
        return
    except ImportError:
        pass
    mod = types.ModuleType("resource")
    mod.RUSAGE_SELF = 0

    def getrusage(_who):
        try:
            import psutil
            peak = psutil.Process(os.getpid()).memory_info().peak_wset // 1024
        except Exception:
            peak = 0
        return types.SimpleNamespace(ru_maxrss=peak)

    mod.getrusage = getrusage
    sys.modules["resource"] = mod


def install_fcntl_shim():
    """kev.suite takes an flock while freezing suites; training never needs it. No-op module on Windows."""
    try:
        import fcntl  # noqa: F401
        return
    except ImportError:
        pass
    mod = types.ModuleType("fcntl")
    mod.LOCK_EX, mod.LOCK_NB, mod.LOCK_UN = 2, 4, 8
    mod.flock = lambda *a, **k: None
    sys.modules["fcntl"] = mod


def set_env_defaults():
    os.environ.setdefault("HF_HOME", "D:/hf_cache")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
