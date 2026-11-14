"""Windows stand-in for the POSIX ``fcntl`` module (only ``flock`` with LOCK_EX/LOCK_NB/LOCK_UN).

Why: benchmarks/controller.py and benchmarks/run.py (protected files) and tests/test_controller.py
``import fcntl``, which does not exist on Windows. Put this folder on PYTHONPATH on Windows only:

  PYTHONPATH="$PWD/code:$PWD/compat_windows"

Behaviour matches what the repo relies on: a non-blocking exclusive lock that raises
BlockingIOError when another handle holds it, released when the file is closed.
"""
import msvcrt
import os

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8


def flock(fd, operation):
    fd = fd if isinstance(fd, int) else fd.fileno()
    position = os.lseek(fd, 0, os.SEEK_CUR)
    os.lseek(fd, 0, os.SEEK_SET)               # lock byte 0, whatever the append position is
    try:
        if operation & LOCK_UN:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        elif operation & LOCK_NB:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
    except OSError as error:
        raise BlockingIOError(error.errno, 'lock is held by another handle') from error
    finally:
        os.lseek(fd, position, os.SEEK_SET)
