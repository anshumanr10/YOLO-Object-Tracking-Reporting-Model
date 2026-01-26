from pathlib import Path
import subprocess

def get_windows_host_ip() -> str:
    result = subprocess.run(
        ["bash", "-lc", "ip route | awk '/default/ {print $3}'"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()

if __name__ == "__main__":
    print(f"Running: {Path(__file__).resolve()}")
    print(get_windows_host_ip())