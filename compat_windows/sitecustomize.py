"""Windows-only: provide the ``os.sysconf`` calls that benchmarks/run.py (protected) makes for RAM size.

Loaded automatically at interpreter start (also in subprocesses) when this folder is on PYTHONPATH.
Only SC_PAGE_SIZE and SC_PHYS_PAGES are supported; anything else raises ValueError like POSIX does
for unknown names.
"""
import os
import sys

if sys.platform == 'win32' and not hasattr(os, 'sysconf'):
    import ctypes

    class _MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    ('total_phys', ctypes.c_ulonglong), ('avail_phys', ctypes.c_ulonglong),
                    ('total_page', ctypes.c_ulonglong), ('avail_page', ctypes.c_ulonglong),
                    ('total_virtual', ctypes.c_ulonglong), ('avail_virtual', ctypes.c_ulonglong),
                    ('avail_ext', ctypes.c_ulonglong)]

    def _sysconf(name):
        if name == 'SC_PAGE_SIZE':
            return 4096
        if name == 'SC_PHYS_PAGES':
            status = _MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError('GlobalMemoryStatusEx failed')
            return status.total_phys // 4096
        raise ValueError(f'unrecognized configuration name {name}')

    os.sysconf = _sysconf
