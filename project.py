# project.py
import asyncio
from camroast.settings import Settings
from camroast.app import CameraApp

if __name__ == "__main__":
    s = Settings()
    app = CameraApp(s)
    asyncio.run(app.run(cam=0))
