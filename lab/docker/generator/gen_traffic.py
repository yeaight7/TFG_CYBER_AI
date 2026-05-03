import argparse
import random
import socket
import time

def tcp_connect(host: str, port: int, timeout: float = 0.2) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
    except Exception:
        pass
    finally:
        try:
            s.close()
        except Exception:
            pass

def http_get(host: str, port: int, timeout: float = 0.5) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        # Connection: close => new TCP flow each request
        req = (
            f"GET /?r={random.randint(0, 10**9)} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            "Connection: close\r\n\r\n"
        )
        s.sendall(req.encode())
        try:
            s.recv(256)
        except Exception:
            pass
    except Exception:
        pass
    finally:
        try:
            s.close()
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="web", help="Docker service name of target (default: web)")
    ap.add_argument("--http-port", type=int, default=80)
    ap.add_argument("--n-http", type=int, default=4000, help="HTTP requests (new connection per request)")
    ap.add_argument("--closed-range", default="30000-30250", help="Closed ports range on target")
    ap.add_argument("--n-closed", type=int, default=2500, help="Connect attempts to closed ports (scan-like)")
    ap.add_argument("--jitter-ms", type=int, default=2, help="Small jitter between actions")
    args = ap.parse_args()

    lo, hi = map(int, args.closed_range.split("-"))
    closed_ports = list(range(lo, hi + 1))
    random.shuffle(closed_ports)

    # 1) “Scan-like”: many short connects to closed ports (RST-heavy)
    for i in range(args.n_closed if False else min(args.n_closed, len(closed_ports))):
        tcp_connect(args.target, closed_ports[i % len(closed_ports)])
        time.sleep(random.random() * args.jitter_ms / 1000)

    # 2) “Benign-ish”: many HTTP requests, each in its own TCP connection
    for _ in range(args.n_http):
        http_get(args.target, args.http_port)
        time.sleep(random.random() * args.jitter_ms / 1000)

if __name__ == "__main__":
    main()
