"""Read-only display process; rendering cannot hold the control Python lock."""
from copy import deepcopy
import multiprocessing as mp
import queue
import threading
import time

import mujoco
import numpy as np


def _display_process(model, shared, done, ready, closed, stats, errors):
    """Draw at a bounded rate; no native viewer thread renders between ticks."""
    window=context=None
    try:
        import glfw
        if not glfw.init():raise RuntimeError('无法打开桌面显示，请检查DISPLAY')
        window=glfw.create_window(800,600,'Conveyor: 20 cm/s | pick - bin - return',None,None)
        if window is None:raise RuntimeError('无法创建显示窗口')
        glfw.make_context_current(window);glfw.swap_interval(1)
        # Child display model only. Physics keeps its original model/quality.
        model.vis.quality.offsamples=0;model.vis.quality.shadowsize=512
        data=mujoco.MjData(model);scene=mujoco.MjvScene(model,maxgeom=10000)
        context=mujoco.MjrContext(model,mujoco.mjtFontScale.mjFONTSCALE_100)
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
        camera=mujoco.MjvCamera();camera.lookat[:]=[0.,0.,.5]
        camera.distance=2.4;camera.azimuth=135.;camera.elevation=-25.
        option=mujoco.MjvOption();previous=[None,None]
        def mouse_move(handle,x,y):
            old_x,old_y=previous;previous[:]=[x,y]
            if old_x is None:return
            if glfw.get_mouse_button(handle,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS:
                _,height=glfw.get_window_size(handle)
                mujoco.mjv_moveCamera(model,mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                    (x-old_x)/max(height,1),(y-old_y)/max(height,1),scene,camera)
        def scroll(handle,x,y):
            mujoco.mjv_moveCamera(model,mujoco.mjtMouse.mjMOUSE_ZOOM,0.,-.05*y,scene,camera)
        glfw.set_cursor_pos_callback(window,mouse_move);glfw.set_scroll_callback(window,scroll)
        cpu_started=time.process_time()
        while not done.is_set():
            if glfw.window_should_close(window):closed.set();return
            with shared.get_lock():values=np.frombuffer(shared.get_obj()).copy()
            a=model.nq;b=a+model.nv;c=b+model.nu
            data.qpos[:]=values[:a];data.qvel[:]=values[a:b]
            data.ctrl[:]=values[b:c];data.time=float(values[c])
            model.geom_rgba[:]=values[c+1:].reshape(model.ngeom,4)
            mujoco.mj_forward(model,data)
            mujoco.mjv_updateScene(model,data,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
            width,height=glfw.get_framebuffer_size(window)
            mujoco.mjr_render(mujoco.MjrRect(0,0,width,height),scene,context)
            glfw.swap_buffers(window);glfw.poll_events()
            with stats.get_lock():
                stats[0]+=1;stats[1]=float(data.time);stats[2]=time.process_time()-cpu_started
            ready.set()
            done.wait(1./5.)
    except BaseException as exc:
        errors.put(f'{type(exc).__name__}: {exc}')
    finally:
        if context is not None:context.free()
        if window is not None:glfw.destroy_window(window)
        glfw.terminate()


class PhysicsViewer:
    """One-way snapshots to a private process; GUI edits never become commands."""
    def __init__(self,runner):
        self.runner=runner
        self.model=deepcopy(runner.env.model);self.data=mujoco.MjData(self.model)
        self.context=mp.get_context('spawn')
        self.done=self.context.Event();self.ready=self.context.Event();self.closed=self.context.Event()
        self.shared=self.context.Array('d',self.model.nq+self.model.nv+self.model.nu+1+self.model.ngeom*4)
        self.stats=self.context.Array('d',3);self.errors=self.context.Queue()
        self.thread=None;self.process=None;self.handle=None
        self.closed_by_user=False;self.error=None;self.frames=0;self.last_sim_s=None

    def _copy_snapshot(self):
        # Private parent copy remains useful for tests and initial inspection.
        snap=self.runner.snapshot()
        self.data.qpos[:]=snap['qpos'];self.data.qvel[:]=snap['qvel'];self.data.ctrl[:]=snap['ctrl']
        self.data.time=snap['sim_s'];mujoco.mj_forward(self.model,self.data)
        self.last_sim_s=float(snap['sim_s'])

    def _publish_snapshot(self):
        snap=self.runner.snapshot()
        values=np.r_[snap['qpos'],snap['qvel'],snap['ctrl'],snap['sim_s'],snap['geom_rgba'].ravel()]
        with self.shared.get_lock():np.frombuffer(self.shared.get_obj())[:]=values

    def start(self):
        self._publish_snapshot()
        self.process=self.context.Process(target=_display_process,
            args=(self.model,self.shared,self.done,self.ready,self.closed,self.stats,self.errors),
            name='read-only-physics-display',daemon=True)
        self.process.start()
        deadline=time.monotonic()+15.
        while not self.ready.wait(.05):
            if not self.process.is_alive():raise RuntimeError(f'显示进程启动失败，退出码{self.process.exitcode}')
            if time.monotonic()>deadline:raise RuntimeError('显示进程启动超时')
        self.thread=threading.Thread(target=self._loop,name='display-snapshot-publisher',daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            while not self.done.is_set():
                if self.closed.is_set():
                    self.closed_by_user=True;self.runner.done.set();return
                try:error=self.errors.get_nowait()
                except queue.Empty:error=None
                if error:raise RuntimeError(error)
                if not self.process.is_alive():raise RuntimeError(f'显示进程退出，退出码{self.process.exitcode}')
                self._publish_snapshot()
                with self.stats.get_lock():self.frames=int(self.stats[0]);self.last_sim_s=float(self.stats[1])
                if self.done.wait(1./30.):break
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}';self.runner.done.set()

    def stop(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=5.)
            if self.thread.is_alive():raise RuntimeError('display publisher did not terminate')
        if self.process is not None:
            self.process.join(timeout=5.)
            if self.process.is_alive():
                self.process.terminate();self.process.join(timeout=3.)
                self.error=self.error or 'display process required termination after physics stopped'
        with self.stats.get_lock():self.frames=int(self.stats[0]);self.last_sim_s=float(self.stats[1])

    def summary(self):
        return dict(requested=True,frames=self.frames,last_displayed_sim_s=self.last_sim_s,
            closed_by_user=self.closed_by_user,error=self.error,display_process=True,
            display_cpu_s=float(self.stats[2]),maximum_render_hz=5,
            scope='read-only independent process/model/data; one-way physical snapshots; no control writes')
