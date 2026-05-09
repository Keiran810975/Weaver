import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


def _send(sock_path: str, event: dict) -> None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        raw = json.dumps(event, ensure_ascii=True).encode("utf-8")
        sock.sendto(raw, sock_path)
    except OSError:
        pass
    finally:
        sock.close()


def _run_tool(argv: List[str], timeout: float = 10.0) -> Dict[str, object]:
    exe = shutil.which(argv[0])
    if not exe:
        return {"available": False, "tool": argv[0], "stdout": "", "stderr": "not found"}
    try:
        proc = subprocess.run(
            [exe, *argv[1:]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "available": True,
            "tool": exe,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        return {"available": True, "tool": exe, "stdout": "", "stderr": str(exc)}


def _summarize_asm(text: str) -> Dict[str, int]:
    lower = text.lower()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    instruction_lines = [
        line
        for line in lines
        if not line.startswith("//")
        and not line.startswith("#")
        and not line.startswith(".")
        and ";" in line
    ]
    return {
        "lines": len(lines),
        "instructions": len(instruction_lines),
        "global_loads": lower.count("ld.global") + lower.count("global_load"),
        "global_stores": lower.count("st.global") + lower.count("global_store"),
        "shared_ops": lower.count(".shared") + lower.count("lds"),
        "tensor_ops": lower.count("mma") + lower.count("wmma"),
        "barriers": lower.count("bar.sync") + lower.count("barrier"),
        "branches": lower.count(" bra") + lower.count("\tbra") + lower.count("bra.uni"),
        "returns": lower.count("ret;"),
    }


def _extract_ptx(binary: Path) -> Dict[str, object]:
    raw = binary.read_bytes()
    if raw.startswith(b"//") or b".version" in raw[:4096]:
        text = raw.decode("utf-8", errors="ignore")
        return {"available": True, "tool": "raw-ptx", "returncode": 0, "stdout": text, "stderr": ""}
    result = _run_tool(["cuobjdump", "-ptx", str(binary)])
    if result.get("available") and ".version" in str(result.get("stdout", "")):
        return result
    return result


def _extract_sass(binary: Path) -> Dict[str, object]:
    result = _run_tool(["nvdisasm", str(binary)])
    if result.get("available") and str(result.get("stdout", "")).strip():
        return result
    return _run_tool(["cuobjdump", "-sass", str(binary)])


def analyze(binary: Path, kernel: str) -> Dict[str, object]:
    raw = binary.read_bytes()
    digest = hashlib.sha1(raw).hexdigest()

    ptx = _extract_ptx(binary)
    sass = _extract_sass(binary)

    ptx_text = str(ptx.get("stdout", ""))
    sass_text = str(sass.get("stdout", ""))
    ptx_summary = _summarize_asm(ptx_text) if ptx_text else {}
    sass_summary = _summarize_asm(sass_text) if sass_text else {}

    return {
        "kernel": kernel,
        "binary_path": str(binary),
        "binary_sha1": digest,
        "binary_bytes": len(raw),
        "method": "hooked_binary_disassembly",
        "probe_level": "static_warp_block",
        "ptx": {
            "available": bool(ptx.get("available")),
            "tool": ptx.get("tool"),
            "returncode": ptx.get("returncode"),
            "summary": ptx_summary,
            "stderr": str(ptx.get("stderr", ""))[:4000],
        },
        "sass": {
            "available": bool(sass.get("available")),
            "tool": sass.get("tool"),
            "returncode": sass.get("returncode"),
            "summary": sass_summary,
            "stderr": str(sass.get("stderr", ""))[:4000],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Weaver Neutrino-style disassembly sidecar")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--sock", default=os.environ.get("WEAVER_SOCK", "/tmp/weaver.sock"))
    args = parser.parse_args()

    binary = Path(args.binary)
    payload: Dict[str, object]
    try:
        payload = analyze(binary, args.kernel)
        kind = "kernel_disassembly"
    except Exception as exc:
        payload = {
            "kernel": args.kernel,
            "binary_path": str(binary),
            "method": "hooked_binary_disassembly",
            "error": str(exc),
        }
        kind = "kernel_disassembly_error"

    _send(
        args.sock,
        {
            "ts_ns": time.time_ns(),
            "pid": os.getpid(),
            "tid": 0,
            "host": socket.gethostname(),
            "layer": "neutrino",
            "kind": kind,
            "kernel_name": args.kernel,
            "payload": payload,
        },
    )


if __name__ == "__main__":
    main()
