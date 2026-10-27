"""Windows Job Object that kills every assigned child process when this
program exits, even on a crash, so llama-server can never be orphaned
holding VRAM. On Linux the kernel does the same through PR_SET_PDEATHSIG
(see child_options)."""
import ctypes
import signal
import subprocess
import sys
from ctypes import wintypes

_job = None


def _die_with_parent():
    """Runs in the child before it starts (Linux): SIGTERM it when we exit."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)   # 1 = PR_SET_PDEATHSIG
    except OSError:
        pass


def child_options():
    """Extra arguments for starting a helper process: no console window on Windows,
    and on Linux the child is killed when this program exits."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    if sys.platform.startswith("linux"):
        return {"preexec_fn": _die_with_parent}
    return {}


def _create_job():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC_LIMITS(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED_LIMITS(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC_LIMITS), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32.CreateJobObjectW.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = EXTENDED_LIMITS()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # ExtendedLimitInformation
    return job if ok else None


def attach(pid):
    """Tie process `pid` to our lifetime. No-op outside Windows."""
    global _job
    if sys.platform != "win32":
        return False
    if _job is None:
        _job = _create_job()
    if not _job:
        return False
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    h = k32.OpenProcess(0x0101, False, pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
    if not h:
        return False
    try:
        return bool(k32.AssignProcessToJobObject(wintypes.HANDLE(_job), wintypes.HANDLE(h)))
    finally:
        k32.CloseHandle(wintypes.HANDLE(h))
