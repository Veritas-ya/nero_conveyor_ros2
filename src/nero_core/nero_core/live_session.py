"""Process-local or panel-owned RAM workspace for replaceable live views."""
from pathlib import Path
from tempfile import mkdtemp
import os
import shutil
import stat


SHARED_MEMORY = Path('/dev/shm')
PANEL_PARENT = 'NERO_PANEL_LIVE_PARENT'


def new_live_parent():
    return Path(mkdtemp(prefix=f'nero-live-{os.getuid()}-{os.getpid()}-', dir=SHARED_MEMORY))


def private_live_parent(path):
    path = Path(path)
    try:
        info = path.lstat()
        return (path.parent == SHARED_MEMORY and path.name.startswith(f'nero-live-{os.getuid()}-')
                and stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o077)
    except OSError:
        return False


def open_live_run(lifetime, *, web_preview):
    parent = os.environ.get(PANEL_PARENT) if web_preview else None
    managed_by_panel = parent is not None
    if parent is None:
        parent = new_live_parent()
        lifetime.callback(shutil.rmtree, parent)
    elif not private_live_parent(parent):
        raise ValueError('网页实时预览需要本用户的私有共享内存目录')
    else:
        parent = Path(parent)
    run = parent/'live'
    run.mkdir(mode=0o700, exist_ok=False)
    if managed_by_panel:
        launcher_pid = os.getppid()
        # The panel normally reads the terminal state and removes its private
        # workspace. The worker still owns its live images if the panel dies first.
        def clean_if_orphaned():
            if os.getppid() != launcher_pid or launcher_pid == 1:
                shutil.rmtree(run, ignore_errors=True)
                try:
                    parent.rmdir()
                except OSError:
                    pass
        lifetime.callback(clean_if_orphaned)
    return run
