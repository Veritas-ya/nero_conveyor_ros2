"""Loopback-only HTTP panel. Starts no hardware until an explicit UI action."""
import argparse
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import signal
from threading import Event, RLock, Thread
import time
from urllib.parse import parse_qs, urlsplit

from .controller import PanelController, PanelError
from .media import ArtifactReader, DemoReader


class PanelApplication:
    def __init__(self, controller):
        self.controller = controller
        self.token = secrets.token_urlsafe(32)
        self.reader = DemoReader() if controller.demo else ArtifactReader()
        self.lock = RLock()
        self.stopped = Event()
        self.state = controller.snapshot()
        self.scene = self.reader.snapshot()
        self.thread = Thread(target=self._monitor, name='panel-read-only-monitor', daemon=True)

    def start(self):
        self.thread.start()

    def _monitor(self):
        while not self.stopped.is_set():
            try:
                self.controller.refresh()
                state = self.controller.snapshot()
                job = state.get('job')
                # Enable/CAN operations retain the last camera/pick visualization.
                run = job['run'] if job and job['kind'] in ('camera', 'pick', 'observe') else self.reader.run
                with self.lock:
                    self.reader.update(run)
                    self.state, self.scene = state, self.reader.snapshot()
            except Exception as error:
                with self.lock:
                    self.state = {**self.state, 'monitor_error': str(error)}
            self.stopped.wait(.2)

    def snapshot(self):
        with self.lock:
            return {**self.state, 'media': {k: self.scene[k] for k in ('revision', 'frame', 'error')},
                'target': {k: self.scene['target'].get(k) for k in ('available', 'point_count', 'sequence', 'version')},
                'plan': {'available': self.scene['plan']['available'],
                         'metadata': self.scene['plan'].get('metadata', {})}}

    def close(self):
        self.controller.close()
        self.stopped.set()
        self.thread.join(timeout=4.)


class PanelHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, application):
        self.application = application
        super().__init__(address, PanelHandler)


class PanelHandler(BaseHTTPRequestHandler):
    server_version = 'NeroLocalPanel/1'

    def log_message(self, *_):
        pass

    def _allowed_host(self):
        host = self.headers.get('Host', '')
        port = self.server.server_address[1]
        return host in (f'127.0.0.1:{port}', f'localhost:{port}')

    def _send(self, code, data=b'', content_type='application/json; charset=utf-8', etag=None):
        if isinstance(data, dict):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy',
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if etag:
            self.send_header('ETag', etag)
        self.end_headers()
        if data:
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self):
        if not self._allowed_host():
            self._send(403, {'error': '仅允许通过本机地址访问'})
            return
        app = self.server.application
        path = urlsplit(self.path).path
        if path == '/api/bootstrap':
            self._send(200, {'token': app.token, 'demo': app.controller.demo,
                'initial_options': app.controller.operator_options(),
                'presets': app.controller.presets(), 'workspace': str(app.controller.root)})
        elif path == '/api/state':
            self._send(200, app.snapshot())
        elif path == '/api/scene':
            with app.lock:
                scene = app.scene
            etag = '"'+hashlib.sha256(scene['revision'].encode()).hexdigest()[:20]+'"'
            if self.headers.get('If-None-Match') == etag:
                self._send(304, etag=etag)
            else:
                self._send(200, scene, etag=etag)
        elif path in ('/api/image/live.jpg', '/api/image/frozen.jpg'):
            with app.lock:
                live = path.endswith('live.jpg')
                version = app.scene['frame' if live else 'target'].get('version')
                requested = parse_qs(urlsplit(self.path).query).get('v', [None])[0]
                if requested is not None and requested != str(version):
                    self._send(409, {'error': '画面版本已更新，请读取新的曝光编号'})
                    return
                data = app.reader.live_image if live else app.reader.frozen_image
            self._send(200 if data else 404, data or b'', 'image/jpeg')
        elif path == '/favicon.ico':
            self._send(204)
        elif path in ('/', '/index.html', '/app.css', '/app.js'):
            name = 'index.html' if path == '/' else path[1:]
            mime = {'index.html': 'text/html', 'app.css': 'text/css', 'app.js': 'text/javascript'}[name]
            self._send(200, (Path(__file__).parent/'web'/name).read_bytes(), mime+'; charset=utf-8')
        else:
            self._send(404, {'error': '接口不存在'})

    def do_POST(self):
        app = self.server.application
        port = self.server.server_address[1]
        origin = self.headers.get('Origin')
        if (not self._allowed_host() or origin is not None and origin not in
                (f'http://127.0.0.1:{port}', f'http://localhost:{port}') or
                not secrets.compare_digest(self.headers.get('X-Control-Token', ''), app.token)):
            self._send(403, {'error': '本地操作令牌或来源无效，请刷新页面'})
            return
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            self._send(415, {'error': '操作需要 JSON 参数'})
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 16384:
                raise PanelError('操作参数长度无效')
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict) or set(request)-{'action', 'options'}:
                raise PanelError('操作参数无效')
            if urlsplit(self.path).path != '/api/action':
                self._send(404, {'error': '接口不存在'})
                return
            action = request.get('action')
            if action == 'stop':
                app.controller.stop()
            elif action == 'rearm':
                app.controller.rearm()
            else:
                app.controller.start(action, request.get('options', {}))
            self._send(202, {'accepted': True})
        except (PanelError, ValueError, TypeError) as error:
            self._send(409, {'error': str(error)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--demo', action='store_true', help='仅展示合成演示，不调用 CAN、相机或机械臂')
    parser.add_argument('--observe-run', type=Path, help='只读接入本工作区已有运行，不启动设备')
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error('端口无效')
    root = args.workspace.resolve()
    directory = root/'runs/control_panel'
    directory.mkdir(parents=True, exist_ok=True)
    ownership = (directory/(f'.demo-{args.port}.lock' if args.demo else '.server.lock')).open('a')
    try:
        fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('已有本地控制界面正在运行，请打开原页面')
    controller = PanelController(root, demo=args.demo)
    try:
        controller.presets()
        if args.observe_run:
            if args.demo:
                raise ValueError('演示与真实运行查看不能同时启用')
            controller.observe(args.observe_run)
    except (OSError, ValueError) as error:
        parser.error('无法加载工作区配置：'+str(error))
    app = PanelApplication(controller)
    try:
        server = PanelHTTPServer(('127.0.0.1', args.port), app)
    except OSError as error:
        parser.error('界面端口不可用：'+str(error))
    app.start()
    closing = Event()
    def request_close(*_):
        if not closing.is_set():
            closing.set()
            Thread(target=server.shutdown, daemon=True).start()
    previous = {number: signal.signal(number, request_close) for number in (signal.SIGINT, signal.SIGTERM)}
    print(f'本地操作界面：http://127.0.0.1:{server.server_address[1]}', flush=True)
    print('演示模式：所有设备操作均为模拟。' if args.demo else
          '等待网页操作；尚未启动 CAN、使能机械臂或打开相机。', flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        try:
            app.close()
        finally:
            server.server_close()
            ownership.close()
            for number, handler in previous.items():
                signal.signal(number, handler)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
