from __future__ import annotations

import enum
import queue
import sys
import threading
import time
from fractions import Fraction
from typing import NamedTuple

import click
import cv2
import numpy as np
from typing_extensions import Self

# --- ライブラリのインポートとエラーチェック ---
try:
    import sounddevice as sd
except ImportError:
    print("Error: sounddevice is not installed. Please run 'pip install sounddevice numpy'")
    sys.exit(1)

try:
    from cyndilib.sender import Sender
    from cyndilib.video_frame import VideoSendFrame
    from cyndilib.audio_frame import AudioSendFrame
    from cyndilib.wrapper.ndi_structs import FourCC
except ImportError:
    print("Error: cyndilib is not installed. Please run 'pip install cyndilib'")
    sys.exit(1)


class PixFmt(enum.Enum):
    """NDIのピクセルフォーマットとOpenCVの色変換コードを対応させるEnum"""
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
    """コマンドラインオプションを保持するデータクラス"""
    pix_fmt: PixFmt
    xres: int
    yres: int
    fps: float
    video_device: int
    sender_name: str
    no_audio: bool
    audio_device: int | None
    sample_rate: int
    audio_channels: int


class VideoCaptureThread(threading.Thread):
    """カメラから映像をキャプチャし、生のフレームをキューに入れるスレッド"""
    def __init__(self, device_index: int, width: int, height: int, fps: float, out_queue: queue.Queue):
        super().__init__()
        self.out_queue = out_queue
        self.running = True
        self.daemon = True

        self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video device: {device_index}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
        
        self.actual_xres = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_yres = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)


    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            if self.out_queue.full():
                try:
                    self.out_queue.get_nowait()
                except queue.Empty:
                    pass
            self.out_queue.put(frame)

    def stop(self):
        self.running = False
        self.join(timeout=2)
        self.cap.release()

class VideoProcessThread(threading.Thread):
    """生の映像フレームを処理（色変換）し、別のキューに入れるスレッド"""
    def __init__(self, in_queue: queue.Queue, out_queue: queue.Queue, pix_fmt: PixFmt):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.pix_fmt = pix_fmt
        self.running = True
        self.daemon = True

    def run(self):
        while self.running:
            try:
                frame = self.in_queue.get(timeout=1.0)
                
                if self.pix_fmt.cv_color_code is not None:
                    processed_frame = cv2.cvtColor(frame, self.pix_fmt.cv_color_code)
                else:
                    b, g, r = cv2.split(frame)
                    alpha = np.full(b.shape, 255, dtype=b.dtype)
                    processed_frame = cv2.merge((b, g, r, alpha))
                
                if self.out_queue.full():
                    try:
                        self.out_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.out_queue.put(processed_frame)

            except queue.Empty:
                continue

    def stop(self):
        self.running = False
        self.join(timeout=2)

class SendThread(threading.Thread):
    """送信キューからデータを取り出し、NDI送信するスレッド"""
    def __init__(self, sender: Sender, in_queue: queue.Queue, has_audio: bool):
        super().__init__()
        self.sender = sender
        self.in_queue = in_queue
        self.has_audio = has_audio
        self.running = True
        self.daemon = True

    def run(self):
        while self.running:
            try:
                item = self.in_queue.get(timeout=1.0)
                if self.has_audio:
                    video_frame, audio_frame = item
                    self.sender.write_video_and_audio(video_frame.ravel(), audio_frame)
                else:
                    self.sender.write_video_async(item.ravel())
            except queue.Empty:
                continue
    
    def stop(self):
        self.running = False
        self.join(timeout=2)


def parse_frame_rate(fr_str: str) -> Fraction:
    """文字列からフレームレートをFractionオブジェクトに変換する"""
    common_rates = {"23.98": Fraction(24000, 1001), "29.97": Fraction(30000, 1001), "59.94": Fraction(60000, 1001)}
    if fr_str in common_rates: return common_rates[fr_str]
    if "/" in fr_str: n, d = [int(s) for s in fr_str.split("/")]; return Fraction(n, d)
    if "." in fr_str: return Fraction(fr_str).limit_denominator(2000)
    return Fraction(int(fr_str), 1)


def capture_and_send(opts: Options) -> None:
    """各処理スレッドを管理し、NDIストリームを送信する"""
    
    raw_video_queue = queue.Queue(maxsize=2)
    processed_video_queue = queue.Queue(maxsize=2)
    send_queue = queue.Queue(maxsize=2)
    
    try:
        video_capture_thread = VideoCaptureThread(
            opts.video_device, opts.xres, opts.yres, opts.fps, raw_video_queue
        )
    except IOError as e:
        print(f"Fatal: Could not initialize video capture thread. Error: {e}", file=sys.stderr)
        return

    video_process_thread = VideoProcessThread(
        raw_video_queue, processed_video_queue, opts.pix_fmt
    )

    actual_xres = video_capture_thread.actual_xres
    actual_yres = video_capture_thread.actual_yres
    actual_fps = video_capture_thread.actual_fps
    print(f"Camera opened with settings: {actual_xres}x{actual_yres} @ {actual_fps:.2f} FPS")

    sender = Sender(opts.sender_name)
    vf = VideoSendFrame()
    vf.set_resolution(actual_xres, actual_yres)
    vf.set_frame_rate(Fraction(actual_fps).limit_denominator())
    vf.set_fourcc(opts.pix_fmt.four_cc)
    sender.set_video_frame(vf)

    audio_stream = None
    audio_queue = None

    if not opts.no_audio:
        audio_queue = queue.Queue(maxsize=10)
        if actual_fps == 0:
            raise ValueError("Actual FPS from camera is 0, cannot calculate audio samples per frame.")
        samples_per_frame = round(opts.sample_rate / actual_fps)
        
        af = AudioSendFrame()
        af.sample_rate = opts.sample_rate
        af.num_channels = opts.audio_channels
        af.set_max_num_samples(samples_per_frame)
        sender.set_audio_frame(af)
        
        def audio_callback(indata, frames, time, status):
            if status:
                if 'input overflow' not in str(status):
                    print(f"Audio callback status: {status}", file=sys.stderr)
            if audio_queue:
                if audio_queue.full():
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                audio_queue.put(indata.copy().T)

        audio_stream = sd.InputStream(
            device=opts.audio_device, samplerate=opts.sample_rate,
            channels=opts.audio_channels, callback=audio_callback,
            blocksize=samples_per_frame, dtype='float32',
            latency='low'
        )

    send_thread = SendThread(sender, send_queue, has_audio=not opts.no_audio)

    video_capture_thread.start()
    video_process_thread.start()
    if audio_stream:
        audio_stream.start()
    send_thread.start()

    try:
        with sender:
            print("Starting assembler loop...")
            if not opts.no_audio and audio_queue is not None:
                last_good_video_frame = None
                try:
                    last_good_video_frame = processed_video_queue.get(timeout=5.0)
                    audio_queue.get(timeout=5.0)
                    print("Initial audio and video frames received. Starting stream.")
                except queue.Empty:
                    raise RuntimeError("Failed to receive initial audio or video frame within 5 seconds.")

                while True:
                    try:
                        audio_data = audio_queue.get(timeout=2.0)
                        
                        try:
                            while True:
                                last_good_video_frame = processed_video_queue.get_nowait()
                        except queue.Empty:
                            pass
                        
                        if send_queue.full():
                            try:
                                send_queue.get_nowait()
                            except queue.Empty:
                                pass
                        send_queue.put((last_good_video_frame, audio_data))

                    except queue.Empty:
                        print("Error: Audio queue was empty for 2 seconds. Stopping stream.", file=sys.stderr)
                        break
            else:
                while True:
                    try:
                        frame = processed_video_queue.get(timeout=2.0)
                        if send_queue.full():
                            try:
                                send_queue.get_nowait()
                            except queue.Empty:
                                pass
                        send_queue.put(frame)
                    except queue.Empty:
                        print("Error: Processed video queue was empty for 2 seconds. Stopping stream.", file=sys.stderr)
                        break
    finally:
        print("\nStopping stream resources...")
        send_thread.stop()
        if audio_stream:
            audio_stream.stop()
            audio_stream.close()
        video_process_thread.stop()
        video_capture_thread.stop()
        print("Stream stopped.")


@click.command()
@click.option('--list-devices', is_flag=True, help='List available video and audio devices and exit.')
@click.option('--no-audio', is_flag=True, help='Disable audio and send video only.')
@click.option('--pix-fmt', type=click.Choice([m.name for m in PixFmt], case_sensitive=False), default=PixFmt.BGR.name, show_default=True, help='Pixel format.')
@click.option('-x', '--x-res', type=int, default=1920, show_default=True, help='Horizontal resolution.')
@click.option('-y', '--y-res', type=int, default=1080, show_default=True, help='Vertical resolution.')
@click.option('--fps', type=str, default='30', show_default=True, help='Frame rate (e.g., 30, 29.97, 60000/1001).')
@click.option('-d', '--video-device', type=int, default=0, show_default=True, help='Video device index.')
@click.option('--audio-device', type=int, default=None, show_default=False, help='Audio device ID (ignored if --no-audio).')
@click.option('--sample-rate', type=int, default=48000, show_default=True, help='Audio sample rate (ignored if --no-audio).')
@click.option('--audio-channels', type=int, default=2, show_default=True, help='Number of audio channels (ignored if --no-audio).')
@click.option('-n', '--sender-name', type=str, default='TX', show_default=True, help='NDI name for the sender.')
def main(list_devices: bool, no_audio: bool, pix_fmt: str, x_res: int, y_res: int, fps: str, video_device: int, audio_device: int, sample_rate: int, audio_channels: int, sender_name: str):
    """Captures video and/or audio from devices and sends them as a low-latency NDI stream."""
    if list_devices:
        print("--- Available Video Devices (OpenCV) ---")
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"  Device {i}: Available")
                cap.release()
            else:
                break
        print("\n--- Available Audio Devices (sounddevice) ---")
        print(sd.query_devices())
        return

    try:
        frame_rate = float(parse_frame_rate(fps))
        opts = Options(
            pix_fmt=PixFmt.from_str(pix_fmt), xres=x_res, yres=y_res,
            fps=frame_rate, video_device=video_device, sender_name=sender_name,
            no_audio=no_audio, audio_device=audio_device,
            sample_rate=sample_rate, audio_channels=audio_channels
        )
        capture_and_send(opts)
    except (IOError, ValueError, RuntimeError) as e:
        print(f"An error occurred: {type(e).__name__}: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nStream stopped by user.")


if __name__ == '__main__':
    main()
