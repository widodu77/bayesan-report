"""Download the Lichess games file from Kaggle into data/raw/.

Source: https://www.kaggle.com/datasets/datasnaek/chess (CC0: Public Domain).
The file is shipped with the code, so this script does nothing if it is present.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/datasnaek/chess"
TARGET = Path(__file__).resolve().parent / "data" / "raw" / "games.csv"


def main():
    if TARGET.exists():
        print(f"{TARGET.name} already present, nothing to do")
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(request, timeout=120) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        TARGET.write_bytes(zf.read("games.csv"))
    print(f"saved {TARGET} ({TARGET.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
