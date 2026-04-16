import time

from weaver.collector import enable_python_collector


def work(n: int) -> int:
    s = 0
    for i in range(n):
        s += i * i
    return s


if __name__ == "__main__":
    enable_python_collector(socket_path="/tmp/weaver.sock", sample_rate=1)
    for _ in range(3):
        work(50000)
        time.sleep(0.1)
