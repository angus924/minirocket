"""Process-lifetime peaks, including native allocations and compression."""
import ctypes
import os
import resource
import sys


class _RusageV4(ctypes.Structure):
    # Public Apple sys/resource.h layout: UUID followed by35 uint64 fields.
    _fields_ = [('uuid', ctypes.c_uint8 * 16), ('values', ctypes.c_uint64 * 35)]


def snapshot():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = int(value if sys.platform == 'darwin' else value * 1024)
    physical = None
    if sys.platform == 'darwin':
        library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        function = library.proc_pid_rusage
        function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        function.restype = ctypes.c_int
        info = _RusageV4()
        if function(os.getpid(), 4, ctypes.byref(info)) != 0:
            raise OSError(ctypes.get_errno(), 'proc_pid_rusage failed')
        physical = int(info.values[28])  # ri_lifetime_max_phys_footprint
        if physical <= 0:
            raise RuntimeError('macOS did not provide a lifetime physical-footprint peak')
    return dict(peak_rss_bytes=rss, peak_physical_footprint_bytes=physical,
                peak_memory_bytes=physical if physical is not None else rss,
                peak_memory_metric='macos_peak_physical_footprint' if physical is not None else 'peak_rss')
