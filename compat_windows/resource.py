"""Windows stand-in for the POSIX ``resource`` module: only getrusage(RUSAGE_SELF).ru_maxrss.

benchmarks/memory.py (protected) reads ru_maxrss in KiB on non-macOS platforms. Here it is the process's
peak working set (Windows' closest equivalent of peak RSS). Memory numbers measured this way are not
comparable with the macOS physical-footprint numbers in the published results (conflict C9 in
README_GROUP_PRUNING.md).
"""
import ctypes
from ctypes import wintypes

RUSAGE_SELF = 0


class _Counters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


class _Usage:
    def __init__(self, ru_maxrss):
        self.ru_maxrss = ru_maxrss


def getrusage(who):
    if who != RUSAGE_SELF:
        raise ValueError('only RUSAGE_SELF is supported')
    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), 'GetProcessMemoryInfo failed')
    return _Usage(counters.PeakWorkingSetSize // 1024)       # KiB, as on Linux
