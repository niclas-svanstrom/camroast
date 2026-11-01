# camroast/jokelog.py
"""A time log of the shows, one text file per day in LOG_DIR, to look back at the jokes afterwards."""
import os
import time


def write(folder: str, head: str, *lines: str):
    """Append one entry: the time and head on the first line, the other lines indented under it."""
    now = time.localtime()
    entry = [f"{time.strftime('%H:%M:%S', now)}  {head}"] + [f"          {line}" for line in lines]
    try:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, time.strftime("skamt-%Y-%m-%d.txt", now)), "a", encoding="utf-8") as f:
            f.write("\n".join(entry) + "\n\n")
    except OSError as e:
        print(f"joke log: {e}")
