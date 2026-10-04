# project.py
import asyncio
import os
import sys

# Torch's OpenMP workers must sleep instead of spin between ops, or they starve the camera
# loop and the audio threads. Must be set before torch is imported.
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from camroast.settings import Settings  # noqa: E402
from camroast.app import CameraApp  # noqa: E402

if __name__ == "__main__":
    # Swedish characters in the console regardless of the code page
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    s = Settings()
    app = CameraApp(s)
    asyncio.run(app.run(cam=s.camera_source))
