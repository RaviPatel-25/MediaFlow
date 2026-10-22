import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import imageio_ffmpeg
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
MAX_VIDEO_MINUTES = int(os.getenv("MAX_VIDEO_MINUTES", "60"))
ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com"}
RESOLUTIONS = [144, 240, 360, 480, 720, 1080, 1440, 2160]
job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

app = FastAPI(title="Video Downloader", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class URLRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)


class DownloadRequest(URLRequest):
    height: int = Field(default=480, ge=0, le=2160)


def validate_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise HTTPException(status_code=400, detail="Enter a valid https:// video URL.")
    host = parsed.hostname.lower()
    if host not in ALLOWED_HOSTS:
        raise HTTPException(
            status_code=400,
            detail="This demo currently accepts YouTube and youtu.be URLs only."
        )
    return url


def run_ytdlp(args, timeout=180):
    command = [
        sys.executable, "-m", "yt_dlp",
        "--ffmpeg-location", FFMPEG_EXE,
        "--no-playlist",
        "--no-warnings",
        *args
    ]
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="The request timed out. Try a shorter video.")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not start downloader: {exc}")


@app.get("/", response_class=HTMLResponse)
def home():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "ffmpeg_available": Path(FFMPEG_EXE).exists()}


@app.post("/api/info")
async def video_info(payload: URLRequest):
    url = validate_url(payload.url)
    async with job_semaphore:
        result = run_ytdlp(["--dump-single-json", "--skip-download", url], timeout=90)
    if result.returncode != 0:
        message = (result.stderr or "Unable to read video information.")[-1200:]
        raise HTTPException(status_code=422, detail=message)

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="The video service returned unreadable information.")

    formats = info.get("formats") or []
    heights = sorted({
        int(item["height"]) for item in formats
        if item.get("height") and item.get("vcodec") != "none"
    })
    options = []
    for height in RESOLUTIONS:
        if any(h <= height for h in heights):
            options.append({
                "height": height,
                "label": f"{height}p",
                "exact": height in heights
            })

    if not options:
        options = [{"height": 0, "label": "Best available", "exact": True}]

    duration = info.get("duration")
    if duration and duration > MAX_VIDEO_MINUTES * 60:
        raise HTTPException(
            status_code=413,
            detail=f"This host demo limits videos to {MAX_VIDEO_MINUTES} minutes."
        )

    return {
        "title": info.get("title") or "Video",
        "duration": duration,
        "thumbnail": info.get("thumbnail"),
        "options": options,
        "best_available": True
    }


@app.post("/api/download")
async def download(payload: DownloadRequest):
    url = validate_url(payload.url)
    height = payload.height
    temp_dir = tempfile.mkdtemp(prefix="video-download-")
    output_template = str(Path(temp_dir) / "%(title).100B [%(id)s].%(ext)s")

    if height == 0:
        selector = "bestvideo+bestaudio/best"
    else:
        selector = f"bestvideo[height<={height}]+bestaudio/best[height<={height}]"

    args = [
        "-f", selector,
        "--merge-output-format", "mp4",
        "--remux-video", "mp4",
        "-o", output_template,
        "--print", "after_move:filepath",
        url
    ]

    try:
        async with job_semaphore:
            result = await asyncio.to_thread(run_ytdlp, args, 900)

        if result.returncode != 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            message = (result.stderr or "Download or merge failed.")[-1600:]
            raise HTTPException(status_code=422, detail=message)

        # yt-dlp prints the final path after post-processing.
        candidates = []
        printed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if printed:
            last = printed[-1]
            if os.path.isfile(last):
                candidates.append(Path(last))
        if not candidates:
            candidates = [
                p for p in Path(temp_dir).iterdir()
                if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
            ]
        if not candidates:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail="Download finished but no final video file was found.")

        final_file = candidates[-1]
        safe_name = re.sub(r'[^A-Za-z0-9._ ()-]+', "_", final_file.name).strip(" .")
        if not safe_name:
            safe_name = "video.mp4"

        async def cleanup():
            shutil.rmtree(temp_dir, ignore_errors=True)

        # FileResponse sends one merged file; remove its temp directory after response.
        from starlette.background import BackgroundTask
        return FileResponse(
            path=str(final_file),
            media_type="video/mp4",
            filename=safe_name,
            background=BackgroundTask(cleanup)
        )

    except HTTPException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}")
