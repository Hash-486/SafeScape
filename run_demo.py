#!/usr/bin/env python
"""One-command launcher for the SafeScape review demo.

Starts the inference server on every interface, verifies it end to end, and prints
the two URLs you actually need: loopback for the laptop, LAN for a phone.

    .venv/Scripts/python.exe run_demo.py

Why --host 0.0.0.0 and not the uvicorn default: the default binds loopback only, so
the laptop works and every other device on the network is refused. That is the single
most common reason the phone "doesn't work".

Note the phone can reach the API over LAN but the browser UI still cannot use the
microphone there -- getUserMedia is restricted to HTTPS and localhost, so plain
http://<lan-ip> gets no mic. Run the live UI demo on the laptop; see --help.
"""
import argparse
import json
import socket
import subprocess
import tempfile
import sys
import time
from pathlib import Path
from urllib import request as urlrequest

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
CLASSES = ["ambience", "alarm", "horn_skid", "distress_call", "glass_break"]

C = {"ok": "\033[32m", "bad": "\033[31m", "warn": "\033[33m",
     "dim": "\033[90m", "bold": "\033[1m", "off": "\033[0m"}


def paint(tag, text):
    return f"{C[tag]}{text}{C['off']}"


def step(n, text):
    print(f"\n{paint('bold', f'[{n}]')} {text}")


def ok(text):
    print(f"    {paint('ok', 'OK')}   {text}")


def warn(text):
    print(f"    {paint('warn', 'WARN')} {text}")


def fail(text):
    print(f"    {paint('bad', 'FAIL')} {text}")


def lan_ip():
    """Address of the interface that carries default-route traffic.

    Uses a UDP socket, which picks a route without sending anything -- more reliable
    than gethostbyname(), which returns 127.0.0.1 on many Windows setups.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def port_busy(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.6)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def get_json(url, timeout=5):
    with urlrequest.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def post_clip(url, path):
    """Multipart POST without pulling in requests, so this runs on a bare interpreter."""
    boundary = "----safescape-demo-boundary"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="file"; filename="clip.wav"\r\n',
        b"Content-Type: audio/wav\r\n\r\n",
        Path(path).read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urlrequest.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urlrequest.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def firewall_rule_exists(port):
    """Windows only. None means 'could not determine', which is not an error."""
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(
            ["netsh", "advfirewall", "firewall", "show", "rule",
             f"name=SafeScape {port}"],
            capture_output=True, text=True, timeout=15)
        return out.returncode == 0 and "No rules match" not in out.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    ap = argparse.ArgumentParser(
        description="Start and verify the SafeScape demo server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The phone can call the API over LAN, but the browser UI needs a mic,\n"
               "and browsers only grant that on HTTPS or localhost. Demo the UI on\n"
               "the laptop; use the phone to show the API responding if you want.")
    ap.add_argument("--port", type=int, default=8124)
    ap.add_argument("--host", default="0.0.0.0",
                    help="default 0.0.0.0 so phones on the LAN can reach it")
    ap.add_argument("--skip-verify", action="store_true",
                    help="start the server without the per-class check")
    args = ap.parse_args()

    print(paint("bold", "\nSafeScape demo launcher"))
    print(paint("dim", f"  {ROOT}"))

    # ---- 1. preflight -------------------------------------------------------
    step(1, "Checking the exported model bundles")
    for key in ("mfcc_cnn", "logmel_crnn", "transformer"):
        bundle = ROOT / "models" / "exported" / key
        missing = [f for f in ("best_model.pt", "label_map.json", "preprocess_config.json")
                   if not (bundle / f).exists()]
        if missing:
            fail(f"missing from models/exported/{key}: {', '.join(missing)}")
            print("\n    Re-export with: bash scripts/run_v2_all_models.sh")
            return 1
        cfg = json.loads((bundle / "preprocess_config.json").read_text())
        ok(f"{key:<12} quantized={cfg.get('quantized', False)}, "
           f"{(bundle / 'best_model.pt').stat().st_size // 1024} KB"
           + ("" if (bundle / "calibration.json").exists() else "  (no calibration.json)"))

    # ---- 2. port ------------------------------------------------------------
    step(2, f"Checking port {args.port}")
    if port_busy(args.port):
        fail(f"port {args.port} is already in use")
        print(f"\n    Something is already listening. Either reuse it, or free the port:")
        print(f"      netstat -ano | findstr :{args.port}")
        print(f"      taskkill /PID <pid> /F")
        return 1
    ok("free")

    # ---- 3. firewall --------------------------------------------------------
    step(3, "Checking the inbound firewall rule (for phone access)")
    has_rule = firewall_rule_exists(args.port)
    if has_rule is True:
        ok(f"rule 'SafeScape {args.port}' exists")
    elif has_rule is False:
        warn("no inbound rule -- phones will be blocked by Windows Firewall")
        print(f"\n    Run this once in an {paint('bold', 'Administrator')} terminal:")
        print(paint("dim", f'      netsh advfirewall firewall add rule name="SafeScape {args.port}" '
                           f"dir=in action=allow protocol=TCP localport={args.port}"))
        print("    The laptop demo works regardless; this only affects other devices.")
    else:
        warn("could not check (non-Windows or netsh unavailable)")

    # ---- 4. start -----------------------------------------------------------
    step(4, f"Starting uvicorn on {args.host}:{args.port}")
    # Server output goes to a file, never an unread PIPE: on Windows the pipe buffer
    # fills after ~45 access-log lines and the server freezes mid-demo.
    log_path = Path(tempfile.gettempdir()) / "safescape_server.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "server.app:app",
         "--host", args.host, "--port", str(args.port)],
        cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True)

    health = f"http://127.0.0.1:{args.port}/health"
    for _ in range(40):
        if proc.poll() is not None:
            fail("server exited during startup")
            print(log_path.read_text(errors="ignore"))
            return 1
        try:
            if get_json(health, timeout=2).get("status") == "ok":
                break
        except Exception:
            time.sleep(0.5)
    else:
        fail("server did not become healthy within 20s")
        proc.terminate()
        return 1
    ok(f"healthy -- serving {get_json(health)['model_name']}")

    try:
        # ---- 5. warm + verify -----------------------------------------------
        # held-out test clips (scripts/17_make_ui_testset.py), not training audio
        clips = ROOT / "demo_clips" / "test"
        predict = f"http://127.0.0.1:{args.port}/predict"

        if args.skip_verify:
            warn("verification skipped (--skip-verify)")
        elif not clips.is_dir():
            warn("demo_clips/ not found -- skipping the per-class check")
        else:
            step(5, "Warming the model and verifying every class")
            passed = 0
            checked = 0
            for name in CLASSES:
                wav = clips / f"{name}_1.wav"
                if not wav.exists():
                    warn(f"{name:<14} no clip in demo_clips/")
                    continue
                checked += 1
                try:
                    out = post_clip(predict, wav)
                except Exception as e:
                    fail(f"{name:<14} request failed: {e}")
                    continue
                got, conf = out["predicted_class"], out["confidence"]
                if got == name:
                    passed += 1
                    ok(f"{name:<14} -> {got:<14} {conf * 100:5.1f}%")
                else:
                    fail(f"{name:<14} -> {got:<14} {conf * 100:5.1f}%")
            if checked and passed == checked:
                print(f"\n    {paint('ok', f'All {passed}/{checked} classes correct.')} Model is warm and ready.")
            elif checked:
                print(f"\n    {paint('warn', f'{passed}/{checked} correct.')} "
                      f"Use the live mic carefully; demo_clips/ is your fallback.")

        # ---- 6. the bit you read out loud -----------------------------------
        ip = lan_ip()
        bar = "=" * 58
        print(f"\n{bar}")
        print(paint("bold", "  READY"))
        print(bar)
        print(f"  Laptop UI  {paint('bold', f'http://127.0.0.1:{args.port}/')}")
        print(paint("dim", "             microphone works here (localhost is a secure context)"))
        print(f"  Compare    {paint('bold', f'http://127.0.0.1:{args.port}/compare')}")
        print(paint("dim", "             all three models side by side + UI test run"))
        if ip and args.host == "0.0.0.0":
            print(f"  Phone API  http://{ip}:{args.port}/health")
            print(paint("dim", "             API only -- the browser blocks the mic over plain HTTP"))
        elif args.host != "0.0.0.0":
            print(paint("warn", f"  Bound to {args.host} -- other devices cannot reach this."))
        print(bar)
        print(paint("dim", "  Demo order: ambience, horn_skid, alarm, distress_call, glass_break"))
        print(paint("dim", "  Hold each sound 4-5s, loud and close to the mic."))
        print(paint("dim", "\n  Ctrl+C to stop.\n"))

        proc.wait()
    except KeyboardInterrupt:
        print(paint("dim", "\n\nStopping..."))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(paint("dim", "Server stopped.\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
