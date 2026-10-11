"""Local browser viewer; all simulation and rendering stay on one thread."""

from __future__ import annotations

import json
import math
import struct
import time
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit

import mujoco
import numpy as np

from .controller import ThreeFloorElevatorController
from .viewer import WINDOW_HEIGHT, WINDOW_WIDTH, _make_camera, pick_button


PAGE = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>三层楼房 · MuJoCo</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#111820;color:#e7edf2;font:16px system-ui,sans-serif}
main{max-width:1320px;margin:24px auto;padding:0 20px}header{display:flex;align-items:center;justify-content:space-between;gap:16px}
h1{font-size:24px;margin:0 0 6px}p{color:#b4c2ce;line-height:1.6;margin:8px 0}
canvas{display:block;width:100%;height:auto;background:#090e13;border:1px solid #304354;border-radius:12px;touch-action:none}
.bar{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}.pill,button{background:#1c2a36;border:1px solid #3a4f60;border-radius:8px;padding:9px 14px;color:inherit}
button{cursor:pointer}#message{min-height:24px;color:#edc56d}small{color:#95a9b9}
</style>
<main><header><div><h1>三层楼房与电梯</h1><small>MuJoCo 实时仿真 · 独立机器人测试场景</small></div>
<button id="reset">重置场景</button></header>
<div class="bar"><span class="pill" id="lift">连接中…</span><span class="pill" id="lamps"></span><span class="pill" id="clock"></span></div>
<canvas id="view" width="1280" height="720"></canvas>
<p>左键点实体按钮；右键拖动旋转；滚轮缩放。黄色为墙上灯开关，绿色 ↑ / 红色 ↓ 为电梯呼叫，轿厢按钮标有 1、2、3。侧向按钮可旋转视角后点按。</p>
<p id="message">房间 → 本层候梯平台 → 电梯轿厢。轿厢离站前自动关门、停稳后自动开门；初始停在三层。</p></main>
<script>
const canvas=document.querySelector('#view'),ctx=canvas.getContext('2d'),message=document.querySelector('#message');
async function post(path,body={}){
 const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 if(!r.ok)throw Error(await r.text());return r.json();
}
const showError=e=>{message.textContent='连接失败：'+e.message+'。请确认运行脚本仍在运行。'};
let dragging=false,lastX=0,lastY=0;
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('pointerdown',e=>{
 if(e.button===2){dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);return;}
 if(e.button!==0)return;const r=canvas.getBoundingClientRect();
 post('/click',{x:(e.clientX-r.left)/r.width,y:1-(e.clientY-r.top)/r.height})
 .then(s=>{message.textContent=s.button?'按压：'+s.button:'此处没有按钮，可右键调整视角后重试。'}).catch(showError);
});
canvas.addEventListener('pointermove',e=>{
 if(!dragging)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;
 post('/camera',{dx,dy}).catch(showError);
});
canvas.addEventListener('pointerup',()=>dragging=false);
canvas.addEventListener('pointercancel',()=>dragging=false);
canvas.addEventListener('wheel',e=>{e.preventDefault();post('/camera',{zoom:-Math.sign(e.deltaY)}).catch(showError);},{passive:false});
document.querySelector('#reset').onclick=()=>post('/reset').then(()=>message.textContent='已重置，电梯停在三层。').catch(showError);
async function frame(){
 try{const r=await fetch('/frame.png',{cache:'no-store'});if(!r.ok)throw Error(await r.text());
 const bitmap=await createImageBitmap(await r.blob());ctx.drawImage(bitmap,0,0);bitmap.close();}
 catch(e){showError(e);}setTimeout(frame,40);
}
async function status(){
 try{const r=await fetch('/status');const s=await r.json();
 document.querySelector('#lift').textContent=s.moving?'电梯 → '+s.target+' 层'+(s.closing?' · 关门中':''):'电梯停在 '+s.floor+' 层';
 document.querySelector('#lamps').textContent=s.lamps.map((v,i)=>(i+1)+'层灯 '+(v?'开':'关')).join(' / ');
 document.querySelector('#clock').textContent='仿真 '+s.time.toFixed(1)+' s';}
 catch(e){showError(e);}setTimeout(status,500);
}frame();status();
</script></html>"""


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode an RGB framebuffer without adding an image-library dependency."""
    height, width, channels = rgb.shape
    if channels != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected an uint8 RGB framebuffer")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack("!I", len(payload)) + kind + payload
                + struct.pack("!I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    rows = b"".join(b"\x00" + row.tobytes() for row in rgb)
    header = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows, 1)) + chunk(b"IEND", b"")


def run(model: mujoco.MjModel, data: mujoco.MjData,
        controller: ThreeFloorElevatorController, *, host: str, port: int) -> None:
    """Serve rendered frames and physical picking through a local HTTP server."""
    camera = _make_camera(model)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[3] = 0

    with mujoco.Renderer(model, height=WINDOW_HEIGHT, width=WINDOW_WIDTH) as renderer:
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                pass

            def reply(self, payload: bytes, mime: str, code: int = 200):
                self.send_response(code)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def json_reply(self, payload, code=200):
                self.reply(json.dumps(payload).encode(), "application/json", code)

            def do_GET(self):
                path = urlsplit(self.path).path
                if path == "/":
                    self.reply(PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/frame.png":
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    self.reply(encode_png(renderer.render()), "image/png")
                elif path == "/status":
                    self.json_reply({"floor": controller.current_floor,
                                     "target": controller.current_target,
                                     "moving": controller.motion_active,
                                     "closing": controller.doors_closing,
                                     "time": float(data.time),
                                     "lamps": [bool(model.light_active[controller.lamp_ids[f]]) for f in (1, 2, 3)]})
                else:
                    self.json_reply({"error": "not found"}, 404)

            def do_POST(self):
                origin = self.headers.get("Origin")
                if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                    self.json_reply({"error": "origin mismatch"}, 403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4096:
                        raise ValueError("invalid request size")
                    args = json.loads(self.rfile.read(length))
                    if not isinstance(args, dict):
                        raise ValueError("expected a JSON object")
                    path = urlsplit(self.path).path
                    if path == "/click":
                        x, y = float(args["x"]), float(args["y"])
                        if not (0 <= x <= 1 and 0 <= y <= 1):
                            raise ValueError("click must be inside the viewport")
                        renderer.update_scene(data, camera=camera, scene_option=option)
                        button = pick_button(model, data, renderer.scene, option,
                                             WINDOW_WIDTH / WINDOW_HEIGHT, x, y)
                        if button:
                            controller.mouse_press(button)
                        self.json_reply({"button": button})
                    elif path == "/camera":
                        dx, dy, zoom = (float(args.get(k, 0)) for k in ("dx", "dy", "zoom"))
                        if not all(math.isfinite(v) and abs(v) <= 1000 for v in (dx, dy, zoom)):
                            raise ValueError("invalid camera movement")
                        camera.azimuth = (camera.azimuth + 0.35 * dx) % 360
                        camera.elevation = max(-80., min(10., camera.elevation + 0.3 * dy))
                        camera.distance = max(4., min(32., camera.distance * math.exp(-0.08 * zoom)))
                        self.json_reply({"ok": True})
                    elif path == "/reset":
                        controller.reset()
                        self.json_reply({"ok": True})
                    else:
                        self.json_reply({"error": "not found"}, 404)
                except (KeyError, TypeError, ValueError, OverflowError) as error:
                    self.json_reply({"error": str(error)}, 400)

        class Server(HTTPServer):
            def get_request(self):
                connection, address = super().get_request()
                connection.settimeout(2.)
                return connection, address

            def handle_error(self, _request, _address):
                # Browser navigation can close a frame download mid-response.
                import sys
                if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, TimeoutError)):
                    super().handle_error(_request, _address)

        with Server((host, port), Handler) as server:
            server.timeout = 0.002
            actual_port = server.server_address[1]
            display_host = "127.0.0.1" if host == "0.0.0.0" else host
            print(f"MuJoCo browser viewer: http://{display_host}:{actual_port}/", flush=True)
            print("Press Ctrl+C to stop.", flush=True)
            last_time = time.perf_counter()
            accumulator = 0.0
            try:
                while True:
                    now = time.perf_counter()
                    accumulator = min(accumulator + now - last_time, 0.1)
                    last_time = now
                    while accumulator >= model.opt.timestep:
                        controller.apply_mouse_forces()
                        mujoco.mj_step(model, data)
                        controller.update()
                        accumulator -= model.opt.timestep
                    server.handle_request()
            except KeyboardInterrupt:
                pass
