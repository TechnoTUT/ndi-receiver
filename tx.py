from __future__ import annotations

from typing import NamedTuple, Any, Generator
from typing_extensions import Self
import enum
from fractions import Fraction
from contextlib import contextmanager
import time
import subprocess
import shlex

import click
import cv2
import numpy as np

# pip install cyndilib
try:
    from cyndilib.wrapper.ndi_structs import FourCC
    from cyndilib.video_frame import VideoSendFrame
    from cyndilib.sender import Sender
except ImportError:
    print("Error: cyndilib is not installed. Please run 'pip install cyndilib'")
    exit(1)


class PixFmt(enum.Enum):
    """
    サポートするピクセルフォーマットと、対応するNDIのFourCCをマッピング
    GStreamerから受け取るBGRフォーマットを変換
    """
    RGBA = (FourCC.RGBA, cv2.COLOR_BGR2RGBA)
    BGRA = (FourCC.BGRA, cv2.COLOR_BGR2BGRA)

    def __init__(self, four_cc: FourCC, cv_color_code: int):
        self.four_cc = four_cc
        self.cv_color_code = cv_color_code

    @classmethod
    def from_str(cls, name: str) -> Self:
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unsupported pixel format: {name}")


class Options(NamedTuple):
    pix_fmt: PixFmt
    xres: int
    yres: int
    fps: str
    video_device: str | int
    sender_name: str = 'TX'


def parse_frame_rate(fr_str: str) -> Fraction:
    """
    "30", "29.97", "30000/1001" のような文字列をFractionオブジェクトに変換
    """
    if '/' in fr_str:
        n, d = [int(s) for s in fr_str.split('/')]
    elif '.' in fr_str:
        n = int(float(fr_str) * 1000)
        d = 1001
    else:
        n = int(fr_str)
        d = 1
    return Fraction(n, d)


@contextmanager
def gstreamer_process(opts: Options) -> Generator[subprocess.Popen, Any, None]:
    device_path = f"/dev/video{opts.video_device}"
    fr = parse_frame_rate(opts.fps)

    pipeline = (
        f"gst-launch-1.0 v4l2src device={device_path} io-mode=2 ! "
        f"image/jpeg,width={opts.xres},height={opts.yres},framerate={fr.numerator}/{fr.denominator} ! "
        f"jpegdec ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"fdsink fd=1 sync=false"
    )

    print("--- GStreamer Command ---")
    print(pipeline)
    print("-------------------------")

    proc = subprocess.Popen(
        shlex.split(pipeline), 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        bufsize=0
    )
    
    # GStreamerの起動失敗を即座にチェック
    time.sleep(1) # 念のため、プロセスがエラーを吐き出すのを少し待つ
    if proc.poll() is not None:
        # 起動直後にプロセスが終了した場合、エラーメッセージを読み取って表示
        gst_errors = proc.stderr.read().decode('utf-8', errors='ignore') if proc.stderr else "N/A"
        print("--- GStreamer immediate exit error ---")
        print(gst_errors)
        print("--------------------------------------")
        raise IOError(f"GStreamer process failed to start. Exit code: {proc.returncode}")

    try:
        yield proc
    finally:
        print("Terminating GStreamer process.")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("GStreamer did not terminate gracefully, killing.")
            proc.kill()


def send(opts: Options) -> None:
    """
    GStreamerからパイプで受け取ったフレームをNDIストリームとして送信
    """
    ndi_fr = parse_frame_rate(opts.fps)
    
    sender = Sender(opts.sender_name)
    vf = VideoSendFrame()
    vf.set_resolution(opts.xres, opts.yres)
    vf.set_frame_rate(ndi_fr)
    vf.set_fourcc(opts.pix_fmt.four_cc)
    sender.set_video_frame(vf)

    frame_size_bytes = opts.yres * opts.xres * 3

    with sender:
        with gstreamer_process(opts) as proc:
            stdout = proc.stdout
            stderr = proc.stderr
            if stdout is None or stderr is None:
                raise IOError("Could not get stdout/stderr from GStreamer process.")

            print(f"Sending NDI stream '{opts.sender_name}' at {float(ndi_fr):.2f} FPS...")
            while True:
                buffer = bytearray()
                bytes_left = frame_size_bytes
                
                while bytes_left > 0:
                    chunk = stdout.read(bytes_left)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    bytes_left -= len(chunk)
                
                frame_data = bytes(buffer)

                if len(frame_data) < frame_size_bytes:
                    print("GStreamer stream ended or data incomplete.")
                    gst_errors = stderr.read().decode('utf-8', errors='ignore')
                    if gst_errors:
                        print("--- GStreamer Error Output ---")
                        print(gst_errors.strip())
                        print("------------------------------")
                    break
                
                bgr_frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((opts.yres, opts.xres, 3))
                converted_frame = cv2.cvtColor(bgr_frame, opts.pix_fmt.cv_color_code)
                sender.write_video_async(converted_frame.ravel())


@click.command()
@click.option(
    '--pix-fmt',
    type=click.Choice([m.name for m in PixFmt], case_sensitive=False),
    default=PixFmt.RGBA.name,
    show_default=True,
    help='Pixel format for the NDI stream.',
)
@click.option('-x', '--x-res', type=int, default=1920, show_default=True, help='Horizontal resolution.')
@click.option('-y', '--y-res', type=int, default=1080, show_default=True, help='Vertical resolution.')
@click.option('--fps', type=str, default='30', show_default=True, help='Frame rate (e.g., 30, 60, 30000/1001).')
@click.option(
    '-d', '--video-device',
    type=str,
    default='0',
    show_default=True,
    help='Video device index (e.g., 0 for /dev/video0).',
)
@click.option(
    '-n', '--sender-name',
    type=str,
    default='TX',
    show_default=True,
    help='NDI name for the sender.',
)
def main(pix_fmt: str, x_res: int, y_res: int, fps: str, video_device: str, sender_name: str):
    """
    Captures video from a local webcam using a GStreamer subprocess and sends it as an NDI stream.
    """
    try:
        dev = int(video_device) if video_device.isdigit() else video_device
        opts = Options(
            pix_fmt=PixFmt.from_str(pix_fmt),
            xres=x_res,
            yres=y_res,
            fps=fps,
            video_device=dev,
            sender_name=sender_name,
        )
        send(opts)
    except (IOError, ValueError, KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\nStream stopped by user.")
        else:
            print(f"An error occurred: {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
