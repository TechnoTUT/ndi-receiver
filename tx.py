from __future__ import annotations

import enum
from fractions import Fraction
from typing import NamedTuple
from typing_extensions import Self

import click
import cv2
import numpy as np

try:
    from cyndilib.sender import Sender
    from cyndilib.video_frame import VideoSendFrame
    from cyndilib.wrapper.ndi_structs import FourCC
except ImportError:
    print("Error: cyndilib is not installed. Please run 'pip install cyndilib'")
    exit(1)


class PixFmt(enum.Enum):
    RGBA = (FourCC.RGBA, cv2.COLOR_BGR2RGBA)
    BGRA = (FourCC.BGRA, cv2.COLOR_BGR2BGRA)
    BGR = (FourCC.BGRX, None) 

    def __init__(self, four_cc: FourCC, cv_color_code: int | None):
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
    fps: int
    video_device: int
    sender_name: str


def parse_frame_rate(fr_str: str) -> Fraction:
    common_rates = {
        "23.98": Fraction(24000, 1001),
        "29.97": Fraction(30000, 1001),
        "59.94": Fraction(60000, 1001),
    }
    if fr_str in common_rates:
        return common_rates[fr_str]

    if "/" in fr_str:
        n, d = [int(s) for s in fr_str.split("/")]
        return Fraction(n, d)
    
    if "." in fr_str:
        return Fraction(fr_str).limit_denominator(2000)

    return Fraction(int(fr_str), 1)


def capture_and_send(opts: Options) -> None:
    cap = cv2.VideoCapture(opts.video_device)
    if not cap.isOpened():
        raise IOError(f"Could not open video device: {opts.video_device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, opts.xres)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opts.yres)
    cap.set(cv2.CAP_PROP_FPS, opts.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_xres = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_yres = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened with settings: {actual_xres}x{actual_yres} @ {actual_fps:.2f} FPS")

    if actual_xres != opts.xres or actual_yres != opts.yres:
        print(f"Warning: Camera does not support {opts.xres}x{opts.yres}. Using {actual_xres}x{actual_yres} instead.")

    sender = Sender(opts.sender_name)
    vf = VideoSendFrame()
    vf.set_resolution(actual_xres, actual_yres)
    
    ndi_fr = Fraction(actual_fps).limit_denominator()
    vf.set_frame_rate(ndi_fr)

    vf.set_fourcc(opts.pix_fmt.four_cc)
    sender.set_video_frame(vf)

    print(f"Sending NDI stream '{opts.sender_name}' at {actual_fps:.2f} FPS...")

    with sender:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            # 色変換
            if opts.pix_fmt.cv_color_code is not None:
                converted_frame = cv2.cvtColor(frame, opts.pix_fmt.cv_color_code)
            else:
                b, g, r = cv2.split(frame)
                alpha = np.full(b.shape, 255, dtype=b.dtype)
                converted_frame = cv2.merge((b, g, r, alpha))

            sender.write_video_async(converted_frame.ravel())

    cap.release()
    print("Stream stopped.")


@click.command()
@click.option(
    '--pix-fmt',
    type=click.Choice([m.name for m in PixFmt], case_sensitive=False),
    default=PixFmt.BGR.name,
    show_default=True,
    help='Pixel format for the NDI stream.',
)
@click.option('-x', '--x-res', type=int, default=1920, show_default=True, help='Horizontal resolution.')
@click.option('-y', '--y-res', type=int, default=1080, show_default=True, help='Vertical resolution.')
@click.option('--fps', type=str, default='30', show_default=True, help='Frame rate (e.g., 30, 29.97).')
@click.option(
    '-d', '--video-device',
    type=int,
    default=0,
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
def main(pix_fmt: str, x_res: int, y_res: int, fps: str, video_device: int, sender_name: str):
    """
    Captures video directly from a webcam using OpenCV and sends it as a low-latency NDI stream.
    """
    try:
        frame_rate = int(parse_frame_rate(fps))
        
        opts = Options(
            pix_fmt=PixFmt.from_str(pix_fmt),
            xres=x_res,
            yres=y_res,
            fps=frame_rate,
            video_device=video_device,
            sender_name=sender_name,
        )
        capture_and_send(opts)
    except (IOError, ValueError) as e:
        print(f"An error occurred: {type(e).__name__}: {e}")
    except KeyboardInterrupt:
        print("\nStream stopped by user.")


if __name__ == '__main__':
    main()
