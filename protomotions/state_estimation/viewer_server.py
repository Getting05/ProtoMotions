"""Small loopback-only live dashboard. Physics stays on the main thread."""
import json
import queue
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs


class Dashboard:
    def __init__(self,host,port,motions,metadata,history=1000):
        self.commands=queue.Queue(maxsize=32); self.lock=threading.Lock()
        self.history=deque(maxlen=history); self.frame=b''; self.sequence=0
        self.status=dict(ready=False,message='Starting simulator',**metadata)
        self.motions=motions; dashboard=self
        html=Path(__file__).with_name('viewer.html').read_bytes()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def send(self,code,body,kind='application/json'):
                self.send_response(code);self.send_header('Content-Type',kind)
                self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
                self.end_headers()
                try:self.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError):pass
            def do_GET(self):
                route=urlparse(self.path);params=parse_qs(route.query)
                if route.path=='/':return self.send(200,html,'text/html; charset=utf-8')
                if route.path=='/frame.jpg':
                    with dashboard.lock:frame=dashboard.frame
                    return self.send(200 if frame else 204,frame,'image/jpeg')
                if route.path=='/motions':
                    q=params.get('q',[''])[0].lower()
                    matches=[dict(id=i,name=n) for i,n in enumerate(motions) if q in n.lower() or q==str(i)]
                    return self.send(200,json.dumps(dict(total=len(matches),motions=matches[:250])).encode())
                if route.path=='/state':
                    try:after=int(params.get('after',['-1'])[0])
                    except ValueError:return self.send(400,b'{}')
                    with dashboard.lock:
                        data=dict(status=dashboard.status,sequence=dashboard.sequence,
                            points=[p for p in dashboard.history if p['sequence']>after])
                    return self.send(200,json.dumps(data,allow_nan=False).encode())
                return self.send(404,b'{}')
            def do_POST(self):
                # Reject cross-origin commands to the local service.
                if self.headers.get('Sec-Fetch-Site')=='cross-site':return self.send(403,b'{}')
                origin=self.headers.get('Origin')
                if origin and urlparse(origin).netloc!=self.headers.get('Host'):return self.send(403,b'{}')
                if self.path!='/command':return self.send(404,b'{}')
                try:
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0<size<=2048:raise ValueError('Invalid request length')
                    command=json.loads(self.rfile.read(size))
                    if not isinstance(command,dict):raise ValueError('Expected a command object')
                    if command.get('action') not in ('play','pause','step','reset','motion','placement','speed','stop','loop'):raise ValueError('Unknown action')
                    if command['action']=='motion' and (not isinstance(command.get('id'),int) or not 0<=command['id']<len(motions)):raise ValueError('Motion ID out of range')
                    if command['action']=='placement' and command.get('value') not in ('aligned','drift'):raise ValueError('Invalid placement')
                    if command['action']=='speed' and command.get('value') not in (.25,.5,1.,2.):raise ValueError('Invalid speed')
                    if command['action']=='loop' and not isinstance(command.get('value'),bool):raise ValueError('Invalid loop')
                    dashboard.commands.put_nowait(command)
                except (ValueError,KeyError,TypeError,queue.Full) as exc:
                    return self.send(400,json.dumps(dict(error=str(exc))).encode())
                return self.send(200,b'{"queued":true}')
        self.server=ThreadingHTTPServer((host,port),Handler);self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()

    def update_status(self,**fields):
        with self.lock:self.status={**self.status,**fields}

    def publish(self,point=None,frame=None):
        with self.lock:
            if point is not None:
                self.sequence+=1;self.history.append(dict(point,sequence=self.sequence))
            if frame is not None:self.frame=frame

    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
