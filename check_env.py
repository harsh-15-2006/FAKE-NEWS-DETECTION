"""
SATYA environment check.

Run:  python check_env.py

Prints exactly what the build needs to know: Python version, total RAM,
which Ollama models are pulled, whether the Ollama server is reachable,
and which Python packages are already importable.

Pure stdlib except for an optional `requests` probe (falls back to
urllib if requests is not installed yet).
"""

import json
import platform
import sys
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"


def line(title):
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def check_python():
    line("PYTHON")
    print(f"  version : {sys.version.split()[0]}")
    print(f"  platform: {platform.system()} {platform.release()}")
    print(f"  exe     : {sys.executable}")
    ok = sys.version_info >= (3, 10)
    print(f"  >= 3.10 : {'YES' if ok else 'NO  <-- PROBLEM'}")


def check_ram():
    line("MEMORY")
    total_gb = None
    try:
        if platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_gb = stat.ullTotalPhys / (1024 ** 3)
            avail_gb = stat.ullAvailPhys / (1024 ** 3)
            print(f"  total RAM    : {total_gb:.1f} GB")
            print(f"  available now: {avail_gb:.1f} GB")
        else:
            import os
            total_gb = (
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            ) / (1024 ** 3)
            print(f"  total RAM: {total_gb:.1f} GB")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not read RAM: {exc}")

    if total_gb is not None:
        print()
        if total_gb >= 15:
            print("  VERDICT: qwen3:8b usable (~5.5GB). Best option.")
        elif total_gb >= 7:
            print("  VERDICT: use a 3-4B model (gemma3:4b / llama3.2:3b).")
        else:
            print("  VERDICT: tight. Use the smallest model you have pulled.")


def _get_json(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"__error__": str(exc)}


def check_ollama():
    line("OLLAMA")
    data = _get_json(f"{OLLAMA_URL}/api/tags")
    if "__error__" in data:
        print(f"  server at {OLLAMA_URL}: NOT REACHABLE")
        print(f"  error: {data['__error__']}")
        print("  FIX: open a separate terminal and run:  ollama serve")
        return
    print(f"  server at {OLLAMA_URL}: REACHABLE")
    models = data.get("models", [])
    if not models:
        print("  models pulled: NONE  <-- PROBLEM")
        return
    print(f"  models pulled: {len(models)}")
    for m in models:
        name = m.get("name", "?")
        size = m.get("size", 0) / (1024 ** 3)
        print(f"    - {name:<28} {size:>5.1f} GB")
    print()
    print("  >>> COPY THE MODEL NAME ABOVE AND SEND IT TO ME <<<")


def check_imports():
    line("PYTHON PACKAGES")
    pkgs = [
        ("requests", "core - HTTP"),
        ("yaml", "core - config (package name: PyYAML)"),
        ("rank_bm25", "core - keyword search"),
        ("indic_transliteration", "transliteration"),
        ("streamlit", "dashboard"),
    ]
    for mod, why in pkgs:
        try:
            __import__(mod)
            print(f"  [OK]      {mod:<24} {why}")
        except ImportError:
            print(f"  [MISSING] {mod:<24} {why}")


def check_network():
    line("NETWORK")
    for name, url in [
        ("PyPI", "https://pypi.org/simple/"),
        ("Google Fact Check API host", "https://factchecktools.googleapis.com/"),
    ]:
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=8) as resp:
                print(f"  [OK]   {name}: HTTP {resp.status}")
        except urllib.error.HTTPError as exc:
            # A 4xx from the API host still proves reachability.
            print(f"  [OK]   {name}: reachable (HTTP {exc.code})")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {name}: {exc}")


if __name__ == "__main__":
    print("\nSATYA ENVIRONMENT CHECK")
    check_python()
    check_ram()
    check_ollama()
    check_imports()
    check_network()
    print("\n" + "=" * 58)
    print("Send me: the RAM line, the model names, and any [MISSING]/[FAIL].")
    print("=" * 58 + "\n")
