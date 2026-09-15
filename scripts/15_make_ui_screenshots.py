"""Render the three SafeScape UI screens to PNG with headless Edge.

The live app toggles screens with JS, so each screen is rendered from a static copy of
index.html with the right section pre-activated and realistic content filled in. CSS is
pulled from the running server so the styling is byte-identical to the real app.
"""
import subprocess
from pathlib import Path

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
SERVER = "http://127.0.0.1:8124"
OUT = Path(__file__).parent

HEAD = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SafeScape</title><link rel="stylesheet" href="{SERVER}/static/styles.css"/></head><body>
<header class="topbar"><span class="brand">SafeScape</span>
<span class="conn-dot conn-ok" title="connection status"></span></header><main id="screens">"""

NAV = """</main><nav class="bottom-nav">
<button class="nav-btn {home}"><span class="nav-icon">&#8962;</span><span>Home</span></button>
<button class="nav-btn {hist}"><span class="nav-icon">&#8942;</span><span>History</span></button>
<button class="nav-btn {set}"><span class="nav-icon">&#9881;</span><span>Settings</span></button>
</nav></body></html>"""

HOME = HEAD + """
<section class="screen active">
  <div class="status-card state-hazard">
    <div class="status-icon">&#9673;</div>
    <div class="status-text">Distress call detected</div>
    <div class="status-sub">Confidence 93.6% &middot; alerting trusted contact</div>
  </div>
  <button class="big-button listening">Stop Listening</button>
  <p class="last-result">Last window: distress_call (0.936) &middot; 6.6 ms</p>
</section>""" + NAV.format(home="active", hist="", set="")

HOME_IDLE = HEAD + """
<section class="screen active">
  <div class="status-card state-listening">
    <div class="status-icon">&#9673;</div>
    <div class="status-text">Listening</div>
    <div class="status-sub">Monitoring ambient audio &middot; all clear</div>
  </div>
  <button class="big-button listening">Stop Listening</button>
  <p class="last-result">Last window: ambience (0.972) &middot; 6.4 ms</p>
</section>""" + NAV.format(home="active", hist="", set="")

HISTORY = HEAD + """
<section class="screen active">
  <h2 class="screen-title">Alert History</h2>
  <ul class="history-list">
    <li class="history-item"><div><div class="hi-class">distress_call</div>
      <div class="hi-meta">15/09/2026, 6:04:12 pm</div></div><div class="hi-meta">0.94</div></li>
    <li class="history-item"><div><div class="hi-class">glass_break</div>
      <div class="hi-meta">15/09/2026, 5:52:41 pm</div></div><div class="hi-meta">1.00</div></li>
    <li class="history-item"><div><div class="hi-class">alarm</div>
      <div class="hi-meta">15/09/2026, 5:31:07 pm</div></div><div class="hi-meta">0.87</div></li>
    <li class="history-item"><div><div class="hi-class">horn_skid</div>
      <div class="hi-meta">15/09/2026, 4:58:30 pm</div></div><div class="hi-meta">0.93</div></li>
  </ul>
</section>""" + NAV.format(home="", hist="active", set="")

SETTINGS = HEAD + """
<section class="screen active">
  <h2 class="screen-title">Settings</h2>
  <label class="field-label">Inference server URL</label>
  <input class="text-input" type="text" value="http://10.194.227.126:8000"/>
  <button class="secondary-button">Test Connection</button>
  <p class="test-result ok">Connected &middot; model: logmel_crnn (calibrated)</p>
</section>""" + NAV.format(home="", hist="", set="active")

for name, html in [("ui_home_hazard", HOME), ("ui_home_listening", HOME_IDLE),
                   ("ui_history", HISTORY), ("ui_settings", SETTINGS)]:
    src = OUT / f"{name}.html"
    src.write_text(html, encoding="utf-8")
    png = OUT / f"{name}.png"
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--screenshot={png}", "--window-size=500,900",
                    "--virtual-time-budget=4000", src.as_uri()], check=True, timeout=120)
    print(f"{png.name}: {png.stat().st_size if png.exists() else 'MISSING'} bytes")
