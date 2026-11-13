'use strict';

const $ = id => document.getElementById(id);
const ui = {bootstrap: null, state: null, scene: null, pending: null, tab: 'live',
  liveVersion: null, frozenVersion: null, sceneRevision: null, etag: null,
  logId: 0, followLogs: true, dismissedError: null, polling: false};
const phases = {
  loading_dependencies: '加载程序', model_startup: '准备机器人模型与示教时序',
  can_startup: '连接机械臂反馈', startup_readiness: '检查启动状态与等待位',
  camera_startup: '启动相机', waiting_for_qualified_target: '等待有效工件',
  planning_and_execution: '规划与执行', approach: '接近工件', intercept: '同步抓取',
  lift: '抓取后抬升', transfer: '示教投放', return: '返回等待位',
  taught_carry: '示教投放', taught_return: '示教返回',
};

async function request(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch(path, {...options, signal: controller.signal});
    if (response.status === 304) return {unchanged: true};
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
    return {data, response};
  } finally { clearTimeout(timeout); }
}

function showError(message) {
  if (!message) return;
  $('error-text').textContent = message;
  $('error-banner').classList.remove('hidden');
}

function number(value, digits = 0) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

function setDot(id, value) { $(id).className = `status-dot ${value || ''}`; }

function selectedPreset() {
  return ui.bootstrap?.presets.find(item => item.id === $('preset').value);
}

function settings() {
  return {preset: $('preset').value, belt_speed_cm_s: $('belt-speed').value === '' ? null : Number($('belt-speed').value),
    continuous: $('continuous').checked};
}

function syncSettings() {
  const preset = selectedPreset();
  if (!preset) return;
  const speed = Number($('belt-speed').value);
  $('speed-conversion').textContent = `${number(speed / 100, 3)} m/s`;
  $('belt-speed').max = preset.maximum_speed_cm_s;
  $('advance-value').textContent = number(preset.command_advance_ms);
  $('open-gripper-width').textContent = `${number(preset.open_width_mm)} mm`;
  $('freeze-value').textContent = preset.freeze_upstream_cm === null
    ? '入口上游允许范围内匹配后冻结'
    : `入口上游 ${number(preset.freeze_upstream_cm)}～${number(preset.freeze_upstream_cm - preset.freeze_tolerance_cm)} cm`;
  $('timing-value').textContent = `${preset.approach_s} / ${preset.follow_s} / ${preset.brake_s} s`;
  $('cycle-note').textContent = $('continuous').checked
    ? '每轮示教投放后回等待位，继续下一件。' : '执行一次抓取、示教投放和返回。';
  updateButtons();
}

function updateButtons() {
  const state = ui.state;
  const job = state?.job;
  const active = Boolean(job?.active);
  const pending = Boolean(ui.pending);
  const ready = Boolean(state?.can?.ready);
  const external = Boolean(state?.external_owner_busy);
  $('can-button').disabled = active || external || pending || !ui.bootstrap;
  $('enable-button').disabled = active || external || pending || !ready;
  $('open-gripper-button').disabled = active || external || pending || !ready;
  $('pick-button').disabled = active || external || pending || !ready;
  $('camera-button').disabled = active || external || pending || !ui.bootstrap || $('preset').value === 'tracked';
  $('observe-button').disabled = active || pending || !ui.bootstrap || Boolean(ui.bootstrap.demo);
  $('stop-button').disabled = !active || job?.kind === 'can' || Boolean(job?.stopping) || pending;
  $('rearm-button').disabled = !active || job?.kind !== 'camera' || Boolean(job?.stopping) || pending;
  for (const id of ['preset', 'belt-speed', 'continuous']) $(id).disabled = active || pending;
  for (const [action, id] of Object.entries({can: 'can-button', enable: 'enable-button',
    open_gripper: 'open-gripper-button',
    camera: 'camera-button', pick: 'pick-button', stop: 'stop-button', observe: 'observe-button'})) {
    $(id).classList.toggle('working', ui.pending === action || (active && job.kind === action && ['can', 'enable', 'open_gripper'].includes(action)));
  }
}

function render(state) {
  ui.state = state;
  const job = state.job;
  const active = Boolean(job?.active);
  const can = state.can || {};
  setDot('connection-dot', 'good');
  $('connection-text').textContent = '界面已连接';
  $('can-description').textContent = can.ready ? '1 Mbit/s · 接口已就绪' : can.present ? '接口未就绪' : '未检测到接口';
  $('can-description').title = can.message || '';
  setDot('can-dot', can.ready ? 'good' : '');
  const armChecked = state.arm?.enabled && can.ready;
  $('arm-description').textContent = active && job.kind === 'pick' ? '由抓取程序检查反馈' : armChecked ? '已通过使能与反馈检查' : '等待使能检查';
  setDot('arm-dot', armChecked || (active && job.kind === 'pick') ? 'good' : '');
  if (active && job.options) {
    $('preset').value = job.options.id;
    $('belt-speed').value = job.options.belt_speed_cm_s;
    $('continuous').checked = job.options.continuous;
    syncSettings();
  }
  let title = can.ready ? '设备准备' : '设备待连接';
  let detail = state.external_owner_busy ? '外部程序占用设备；可“查看已有运行”，新任务需先从原终端停止。'
    : '先启动 can0 并使能机械臂，或直接进行仅相机规划。';
  if (state.external_owner_busy) title = '设备由原终端控制';
  if (active) {
    if (job.kind === 'camera') {
      title = '仅相机规划'; detail = state.worker?.status || '正在打开相机与规划模型';
    } else if (job.kind === 'pick' || job.kind === 'observe') {
      const stage = state.progress?.execution_stage || state.progress?.phase;
      title = job.kind === 'observe' ? '只读查看已有运行' : '真机抓取运行中';
      detail = phases[stage] || stage || (job.kind === 'observe' ? '原终端继续控制设备，网页仅展示共享数据。' : '正在启动原抓取程序');
      if (job.kind === 'observe' && state.worker?.status) {
        title = '查看历史运行'; detail = '该运行已结束，当前展示存档画面、点云和规划路径。';
      }
    } else if (job.kind === 'enable') { title = '正在使能机械臂'; detail = '等待七轴使能及真实反馈检查'; }
    else if (job.kind === 'open_gripper') { title = '正在打开夹爪'; detail = '检查机械臂停稳，并等待真实开口反馈'; }
    else { title = '正在启动 can0'; detail = '设置 CAN 接口并检查状态'; }
    if (job.stopping) { title = '正在停止'; detail = '等待原程序结束并释放资源，请稍候。'; }
  } else if (job) {
    title = job.exit_code === 0 ? '任务已结束' : '任务已停止';
    detail = job.kind === 'can' && can.ready ? 'can0 已就绪，可以使能机械臂。'
      : job.kind === 'enable' && armChecked ? '使能检查通过，可以启动抓取。'
      : job.kind === 'open_gripper' && state.gripper?.opened ? `夹爪已打开至 ${number(state.gripper.width_m*1000, 1)} mm。`
      : '运行记录已保留；已有画面与轨迹为本次运行的历史数据。';
  }
  $('session-title').textContent = title;
  $('stop-label').textContent = job?.kind === 'observe' ? '结束查看' : '停止运行';
  $('session-subtitle').textContent = detail;
  $('session-subtitle').title = detail;
  const historical = job?.kind === 'observe' && state.worker?.status;
  setDot('session-dot', active ? (job.stopping || historical ? 'warning' : 'good') : state.error ? 'bad' : can.ready ? 'good' : '');
  // Process elapsed time is supplied as a duration, never inferred across clocks.
  const elapsed = job?.elapsed_s ?? 0;
  $('session-clock').textContent = historical ? 'HISTORY' : active ? `${Math.floor(elapsed/60).toString().padStart(2,'0')}:${Math.floor(elapsed%60).toString().padStart(2,'0')}` : 'STANDBY';
  const target = state.target || {};
  const plan = state.plan || {};
  const metadata = plan.metadata || {};
  $('points-value').textContent = target.available ? target.point_count.toLocaleString('en-US') : '—';
  $('points-caption').textContent = target.available ? `冻结曝光 #${target.sequence}` : '等待有效工件';
  const missed = typeof metadata.actual_remaining_send_lead_s === 'number' && metadata.actual_remaining_send_lead_s < 0;
  $('path-value').textContent = plan.available ? (missed ? '已错过截止' : '路径已生成') : '尚未生成';
  $('path-value').className = `text-value ${plan.available ? missed ? 'warning' : 'good' : ''}`;
  $('path-caption').textContent = plan.available
    ? `${number(metadata.path_duration_s, 2)} s · ${metadata.camera_only ? '模型路径' : '规划输出'}`
    : state.worker?.last_attempt?.error ? '本次规划未通过，查看日志' : '完整规划通过后显示';
  $('planning-value').textContent = plan.available ? number(metadata.planning_wall_s*1000) : '—';
  const lead = metadata.actual_remaining_send_lead_s;
  $('timing-caption').textContent = typeof lead === 'number' ? `生成时截止余量 ${number(lead*1000)} ms` : '本次规划计算';
  const frame = state.media?.frame || {};
  const fresh = active && ['camera', 'pick', 'observe'].includes(job.kind) && frame.available && frame.age_s < 2;
  const cameraText = state.demo ? 'DEMO / SYNTHETIC' : ui.tab === 'frozen' ? 'FROZEN EXPOSURE' : fresh ? 'LIVE / SHARED CAMERA' : frame.available ? 'HISTORY / NOT LIVE' : 'CAMERA OFFLINE';
  $('camera-tag').textContent = cameraText;
  setDot('camera-live-dot', state.demo || fresh ? 'good' : frame.available ? 'warning' : '');
  $('frame-label').textContent = 'FRAME '+(ui.tab === 'frozen' ? target.sequence ?? '—' : frame.sequence ?? '—');
  $('frame-age').textContent = ui.tab === 'frozen' ? '工件原始曝光' : state.demo ? '合成演示画面' : fresh ? `帧龄 ${number(frame.age_s*1000)} ms` : frame.available ? '历史画面 / 已停更' : '预览未开启';
  $('camera-note').textContent = active && job.kind === 'camera' ? '仅相机与模型规划 · CAN 与机器人命令为 0'
    : ui.tab === 'frozen' ? '冻结原始曝光，不叠加其他时刻的工件位置。' : '共享抓取程序的同一相机来源。';
  updateImage(state);
  appendLogs(state.logs || []);
  const recordedError = job?.kind === 'observe' && state.worker?.error ? '原运行记录：'+state.worker.error : state.worker?.error;
  const error = state.error || recordedError || state.monitor_error || state.media?.error;
  if (error && error !== ui.dismissedError) showError(error);
  const path = job?.run;
  $('record-path').textContent = path || '本次运行记录将在启动后自动保存';
  $('record-path').title = path || '';
  $('copy-path').disabled = !path;
  updateButtons();
}

function updateImage(state) {
  const frozen = ui.tab === 'frozen';
  const available = frozen ? state.target?.available : state.media?.frame?.available;
  const version = frozen ? state.target?.version : state.media?.frame?.version;
  const key = `${frozen}:${version}`;
  const image = $('camera-image');
  if (available) {
    if (image.dataset.key !== key) {
      image.dataset.key = key;
      image.src = `/api/image/${frozen ? 'frozen' : 'live'}.jpg?v=${encodeURIComponent(version)}`;
    }
  } else {
    image.hidden = true;
    image.removeAttribute('src');
    image.dataset.key = '';
    $('camera-empty').classList.remove('hidden');
  }
}

function appendLogs(rows) {
  const fresh = rows.filter(row => row.id > ui.logId);
  if (!fresh.length) return;
  $('logs').querySelector('.log-empty')?.remove();
  for (const row of fresh) {
    const line = document.createElement('div');
    line.className = 'log-row'+(/失败|异常|停止\]|拒绝|错误/.test(row.text) ? ' error' : /结果|通过|已就绪/.test(row.text) ? ' result' : '');
    const stamp = document.createElement('span'); stamp.className = 'log-time'; stamp.textContent = row.time;
    const message = document.createElement('span'); message.className = 'log-message'; message.textContent = row.text;
    line.append(stamp, message); $('logs').append(line); ui.logId = row.id;
  }
  while ($('logs').children.length > 160) $('logs').firstElementChild.remove();
  $('log-count').textContent = $('logs').children.length;
  if (ui.followLogs) $('logs').scrollTop = $('logs').scrollHeight;
}

async function action(name) {
  if (ui.pending || !ui.bootstrap) return;
  ui.pending = name; ui.dismissedError = null;
  $('error-banner').classList.add('hidden'); updateButtons();
  try {
    await request('/api/action', {method: 'POST', headers: {'Content-Type': 'application/json',
      'X-Control-Token': ui.bootstrap.token}, body: JSON.stringify({action: name, options: settings()})});
    await poll();
  } catch (error) { showError(error.message); }
  finally { ui.pending = null; updateButtons(); }
}

class CloudView {
  constructor(canvas) {
    this.canvas = canvas; this.context = canvas.getContext('2d');
    this.center = [0, 0, 0]; this.scale = 330; this.yaw = -.55; this.pitch = .86;
    this.scene = {scene: [], target: {points: []}, plan: {tcp: []}}; this.targetKey = null;
    this.drag = null;
    new ResizeObserver(() => this.draw()).observe(canvas);
    canvas.addEventListener('pointerdown', event => {this.drag = [event.clientX, event.clientY]; canvas.setPointerCapture(event.pointerId);});
    canvas.addEventListener('pointerup', () => {this.drag = null;});
    canvas.addEventListener('pointercancel', () => {this.drag = null;});
    canvas.addEventListener('pointermove', event => {
      if (!this.drag) return;
      this.yaw += (event.clientX - this.drag[0]) * .008;
      this.pitch = Math.max(.05, Math.min(1.56, this.pitch + (event.clientY - this.drag[1]) * .008));
      this.drag = [event.clientX, event.clientY]; this.draw();
    });
    canvas.addEventListener('wheel', event => {event.preventDefault(); this.scale = Math.max(30, Math.min(5000, this.scale * Math.exp(-event.deltaY*.001))); this.draw();}, {passive: false});
    canvas.addEventListener('dblclick', () => this.fit());
  }
  setScene(scene) {
    this.scene = scene;
    const token = `${scene.target?.version}:${scene.plan?.available}`;
    if (token !== this.targetKey) {this.targetKey = token; this.fit();} else this.draw();
    $('cloud-empty').classList.toggle('hidden', Boolean(scene.target?.available));
    $('cloud-count').textContent = scene.target?.available ? `${scene.target.point_count.toLocaleString('en-US')} OBJECT POINTS` : scene.scene.length ? 'LIVE SCENE' : '等待点云';
  }
  project(point) {
    const [x,y,z] = point.map((value, i) => value - this.center[i]);
    const c = Math.cos(this.yaw), s = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    return [this.width/2+(c*x-s*y)*this.scale, this.height/2-(s*sp*x+c*sp*y+cp*z)*this.scale];
  }
  line(a, b, color, width = 1) {
    const context = this.context, x = this.project(a), y = this.project(b);
    context.beginPath(); context.moveTo(...x); context.lineTo(...y); context.strokeStyle = color; context.lineWidth = width; context.stroke();
  }
  dots(points, color, size = 1.1) {
    const context = this.context; context.fillStyle = color;
    for (const point of points) {const [x,y] = this.project(point); if(x>=0&&y>=0&&x<this.width&&y<this.height) context.fillRect(x,y,size,size);}
  }
  fit() {
    const important = [...(this.scene.target?.points || []), ...(this.scene.plan?.tcp || []), ...(this.scene.plan?.predicted_points || [])];
    const values = important.length ? important : this.scene.scene || [];
    if (values.length) {
      const low = [0,1,2].map(i => Math.min(...values.map(p => p[i])));
      const high = [0,1,2].map(i => Math.max(...values.map(p => p[i])));
      this.center = low.map((value,i) => (value+high[i])/2);
      this.scale = Math.min(this.canvas.clientWidth, this.canvas.clientHeight)*.76/Math.max(...high.map((v,i)=>v-low[i]), .25);
    }
    this.draw();
  }
  draw() {
    const canvas = this.canvas, context = this.context, ratio = window.devicePixelRatio || 1;
    this.width = canvas.clientWidth; this.height = canvas.clientHeight;
    canvas.width = this.width*ratio; canvas.height = this.height*ratio;
    context.scale(ratio, ratio); context.clearRect(0, 0, this.width, this.height);
    for(let i=-10;i<=10;i++) {
      this.line([i/10,-1,0],[i/10,1,0],'#7994a013', .6);
      this.line([-1,i/10,0],[1,i/10,0],'#7994a013', .6);
    }
    this.line([0,0,0],[.15,0,0],'#bc7f7a80'); this.line([0,0,0],[0,.15,0],'#73a89880'); this.line([0,0,0],[0,0,.15],'#7d9fbd80');
    if ($('show-scene').checked) this.dots(this.scene.scene || [], '#70869a65', 1.1);
    if ($('show-predicted').checked) this.dots(this.scene.plan?.predicted_points || [], '#a3d0a875', 1.3);
    if ($('show-object').checked) {
      this.dots(this.scene.target?.points || [], '#5ad4c6', 1.7);
      const box = this.scene.target?.obb;
      if (box) {
        const signs = []; for(const x of [-1,1]) for(const y of [-1,1]) for(const z of [-1,1]) signs.push([x,y,z]);
        const corners = signs.map(sign => [0,1,2].map(i => box.center_world_m[i]+sign.reduce((sum,value,j)=>sum+box.axes_world_columns[i][j]*value*box.extent_m[j]/2,0)));
        for(let i=0;i<8;i++) for(let j=i+1;j<8;j++) if(signs[i].filter((v,k)=>v!==signs[j][k]).length===1) this.line(corners[i],corners[j],'#76b9a88a');
      }
    }
    if ($('show-path').checked) {
      const path = this.scene.plan?.tcp || [];
      for(let i=1;i<path.length;i++) this.line(path[i-1],path[i],'#e4b66f',2.3);
      if(path.length) {
        const last = this.project(path[path.length-1]); context.beginPath();context.arc(...last,4,0,2*Math.PI);context.fillStyle='#fff1c5';context.fill();
        const first = this.project(path[0]); context.beginPath();context.arc(...first,3,0,2*Math.PI);context.strokeStyle='#e5b770';context.stroke();
        const pose = this.scene.plan.contact_grasp;
        if(pose) {
          const local = [[0,-.05,0],[0,-.05,-.04],[0,.05,-.04],[0,.05,0]];
          const world = local.map(p=>[0,1,2].map(i=>pose[i][3]+p.reduce((sum,v,j)=>sum+pose[i][j]*v,0)));
          for(let i=1;i<world.length;i++) this.line(world[i-1],world[i],'#b0c9dd',1.4);
        }
      }
    }
  }
}

const cloudView = new CloudView($('cloud-canvas'));

async function poll() {
  if (ui.polling || !ui.bootstrap) return;
  ui.polling = true;
  try {
    const {data} = await request('/api/state'); render(data);
    if (data.media?.revision !== ui.sceneRevision) {
      const scene = await request('/api/scene', {headers: ui.etag ? {'If-None-Match': ui.etag} : {}});
      if (!scene.unchanged) {
        ui.scene = scene.data; ui.etag = scene.response.headers.get('ETag');
        ui.sceneRevision = scene.data.revision; cloudView.setScene(scene.data);
      }
    }
  } catch (error) {
    setDot('connection-dot', 'bad'); $('connection-text').textContent = '界面连接中断';
    $('camera-tag').textContent = 'CONNECTION LOST'; setDot('camera-live-dot', 'warning');
    for (const id of ['can-button','enable-button','open-gripper-button','camera-button','pick-button','rearm-button','observe-button']) $(id).disabled=true;
    showError('界面服务连接中断，请检查启动终端。已启动的设备任务不会因页面断线自动终止。');
  } finally { ui.polling = false; }
}

for (const button of document.querySelectorAll('[data-action]')) button.addEventListener('click', () => action(button.dataset.action));
for (const id of ['preset','belt-speed','continuous']) $(id).addEventListener('input', syncSettings);
for (const id of ['show-scene','show-object','show-path','show-predicted']) $(id).addEventListener('change', () => cloudView.draw());
$('fit-button').addEventListener('click', () => cloudView.fit());
$('top-button').addEventListener('click', () => {cloudView.pitch=1.56; cloudView.yaw=0; cloudView.fit();});
$('refresh-button').addEventListener('click', () => poll());
$('dismiss-error').addEventListener('click', () => {ui.dismissedError=$('error-text').textContent; $('error-banner').classList.add('hidden');});
$('follow-log').addEventListener('click', () => {ui.followLogs=!ui.followLogs; $('follow-log').setAttribute('aria-pressed',String(ui.followLogs)); $('follow-log').lastChild.textContent=ui.followLogs?'自动跟随':'跟随已暂停';});
$('copy-path').addEventListener('click', async () => {try {await navigator.clipboard.writeText(ui.state.job.run); $('copy-path').textContent='已复制'; setTimeout(()=>$('copy-path').textContent='复制路径',1500);} catch (error) {showError('复制失败，可从运行记录栏手动选择路径。');}});
for (const tab of ['live','frozen']) $('view-'+tab).addEventListener('click', () => {
  ui.tab=tab;
  for(const name of ['live','frozen']) {$('view-'+name).classList.toggle('selected', name===tab);$('view-'+name).setAttribute('aria-pressed',String(name===tab));}
  $('camera-image').dataset.key=''; $('camera-image').hidden=true; $('camera-empty').classList.remove('hidden');
  if(ui.state) render(ui.state);
});
$('camera-image').addEventListener('load', () => {$('camera-image').hidden=false; $('camera-empty').classList.add('hidden');});
$('camera-image').addEventListener('error', () => {$('camera-image').hidden=true; $('camera-empty').classList.remove('hidden'); $('camera-image').dataset.key='';});
$('footer-date').textContent = new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());

async function boot() {
  try {
    const {data} = await request('/api/bootstrap'); ui.bootstrap=data;
    $('preset').replaceChildren(...data.presets.map(item=>{const option=document.createElement('option');option.value=item.id;option.textContent=item.label;return option;}));
    if(data.initial_options) {
      $('preset').value=data.initial_options.preset;
      $('belt-speed').value=data.initial_options.belt_speed_cm_s;
      $('continuous').checked=data.initial_options.continuous;
    }
    if(data.demo) {$('mode-badge').textContent='离线演示';$('mode-badge').className='badge demo';$('demo-banner').classList.remove('hidden');$('sidebar-mode').textContent='DEMO';}
    syncSettings(); await poll(); setInterval(poll,500);
  } catch(error) {showError('界面初始化失败：'+error.message);}
}
boot();
