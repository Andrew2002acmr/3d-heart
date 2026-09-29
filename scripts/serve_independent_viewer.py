"""Local-only NiiVue check, serving only four explicit cases and the viewer."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    allowed = {f"/data/ct_{case}_{kind}.nii.gz": a.data / f"ct_{case}_{kind}.nii.gz"
               for case in (1001,1002,1003,1004) for kind in ("image","label")}
    allowed["/"] = Path(__file__).with_name("independent_viewer.html")
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = allowed.get(self.path.split("?")[0])
            if path is None or not path.is_file():
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8" if path.suffix == ".html" else "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile)
    print(f"http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
