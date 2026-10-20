"""Screenshots of the demo UI for the report (needs the demo server: start_demo.bat or `python scripts/demo.py server`).

    python docs/report/capture_ui.py [--browser PATH]

Starts a headless Edge/Chrome, opens http://127.0.0.1:8000/ui/, runs the cards that need no webcam
(replay, morph, deepfake, signed verdict, crypto) and saves 2x PNGs to docs/report/figures/ui_*.png.
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
from websockets.sync.client import connect

OUT = Path(__file__).resolve().parent / "figures"
URL = "http://127.0.0.1:8000/ui/"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
PORT = 9333


class Page:
    def __init__(self, ws_url: str):
        self.ws = connect(ws_url, max_size=None)
        self.ids = itertools.count(1)

    def call(self, method: str, **params):
        i = next(self.ids)
        self.ws.send(json.dumps({"id": i, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == i:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"])
        return r["result"].get("value")

    def wait(self, expr: str, timeout: float = 180.0) -> None:
        end = time.time() + timeout
        while time.time() < end:
            if self.js(expr):
                return
            time.sleep(0.5)
        raise TimeoutError(expr)

    def faded(self, selector: str) -> list:
        return self.js(f"""(() => {{ const out = []; let e = document.querySelector({json.dumps(selector)});
            while (e) {{ const s = getComputedStyle(e); if (s.opacity !== '1') out.push([e.tagName, e.id, e.className, s.opacity]); e = e.parentElement; }}
            return out; }})()""")

    def shot(self, selector: str, name: str, pad: int = 8) -> None:
        if self.faded(selector):
            raise RuntimeError(f"{selector} is drawn faded: {self.faded(selector)}")
        box = self.js(f"""(() => {{ const r = document.querySelector({json.dumps(selector)}).getBoundingClientRect();
            return {{x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height}}; }})()""")
        clip = {"x": max(0, box["x"] - pad), "y": max(0, box["y"] - pad), "width": box["w"] + 2 * pad, "height": box["h"] + 2 * pad, "scale": 1}
        data = self.call("Page.captureScreenshot", format="png", clip=clip, captureBeyondViewport=True)["data"]
        (OUT / name).write_bytes(base64.b64decode(data))
        print("wrote", name, f"{int(clip['width'])}x{int(clip['height'])} css px")


def run_card(page: Page, step: str) -> None:
    page.js(f"document.querySelector('#card-{step} button.run').click()")
    page.wait(f"!document.querySelector('#card-{step}').classList.contains('busy') && document.querySelector('#card-{step} .result').innerHTML.length > 0")
    page.wait(f"!document.querySelector('#card-{step} button.run').disabled")  # status refresh finished
    time.sleep(2.0)  # images and transitions inside the result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--browser", default=next((b for b in BROWSERS if Path(b).exists()), None))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="ui-capture-")
    proc = subprocess.Popen([args.browser, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--disable-gpu", "--window-size=1280,900", "about:blank"])
    try:
        for _ in range(60):
            try:
                targets = httpx.get(f"http://127.0.0.1:{PORT}/json/list", timeout=1).json()
                ws = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("browser did not start")
        page = Page(ws)
        page.call("Emulation.setDeviceMetricsOverride", width=1180, height=900, deviceScaleFactor=2, mobile=False)
        page.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
        page.call("Page.enable")
        page.call("Page.navigate", url=URL)
        page.wait("document.querySelector('#status-body .lights') !== null")
        page.shot("header", "ui_header.png", pad=0)
        page.shot("#status", "ui_status.png")
        for step in ("replay", "morph", "deepfake", "verdict", "crypto"):
            run_card(page, step)
            page.shot(f"#card-{step}", f"ui_{step}.png")
        page.js("document.querySelector('.tab[data-tab=results]').click()")
        page.wait("document.querySelector('#results-body table') !== null")
        time.sleep(1.0)
        page.shot("#tab-results", "ui_results.png")
        return 0
    finally:
        proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
