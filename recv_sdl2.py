from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING
from typing_extensions import Self
import enum
import time
import gc
import threading
import queue

import click

# cyndilibの必要なモジュールをインポート
from cyndilib.wrapper.ndi_structs import FourCC
from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
from cyndilib.video_frame import VideoFrameSync
from cyndilib.receiver import Receiver
from cyndilib.finder import Finder
if TYPE_CHECKING:
    from cyndilib.finder import Source

# GUIとレンダリングのためのライブラリをインポート
import sdl2
import sdl2.ext
from OpenGL.GL import *
from OpenGL.GLU import *


class RecvFmt(enum.Enum):
    """受信するピクセルフォーマット（cyndilib.wrapper.ndi_recv.RecvColorFormatの値にマッピング）
    """
    uyvy = RecvColorFormat.UYVY_RGBA    #: UYVY（アルファチャンネルがある場合はRGBA）
    rgb = RecvColorFormat.RGBX_RGBA     #: RGB / RGBA
    bgr = RecvColorFormat.BGRX_BGRA     #: BGR / BGRA

    @classmethod
    def from_str(cls, name: str) -> Self:
        return cls.__members__[name]


class Bandwidth(enum.Enum):
    """受信帯域
    """
    lowest = RecvBandwidth.lowest
    highest = RecvBandwidth.highest

    @classmethod
    def from_str(cls, name: str) -> Self:
        return cls.__members__[name]


class Options(NamedTuple):
    """CLIを通じて設定されるオプション
    """
    sender_name: str = 'ffmpeg_sender'
    """接続するNDIソースの名前"""

    recv_fmt: RecvFmt = RecvFmt.rgb
    """受信ピクセルフォーマット"""

    recv_bandwidth: Bandwidth = Bandwidth.highest
    """受信帯域"""

    fullscreen: bool = False
    """フルスクリーンモードで起動するかどうか"""


def get_source(finder: Finder, name: str) -> Source:
    """Finderを使い、完全な名前またはストリーム名でNDIソースを検索する
    """
    click.echo('NDIソースを待機中...')
    finder.wait_for_sources(10)
    for source in finder:
        if source.name == name or source.stream_name == name:
            return source
    raise Exception(f'ソースが見つかりません。利用可能なソース: {finder.get_source_names()}')


def wait_for_first_frame(receiver: Receiver) -> None:
    """データを含む最初のフレームを受信するまで待機する
    """
    vf = receiver.frame_sync.video_frame
    assert vf is not None
    click.echo('最初のフレームを待機中...')
    while receiver.is_connected():
        receiver.frame_sync.capture_video()
        resolution = vf.get_resolution()
        if min(resolution) > 0 and vf.get_data_size() > 0:
            click.echo('フレームを取得しました。')
            return
        time.sleep(0.01)

def render_texture(frame: bytes, tex_w: int, tex_h: int, win_w: int, win_h: int, texture_id: int, recv_fmt: RecvFmt):
    """受信したフレームデータをOpenGLテクスチャとしてアスペクト比を維持して描画する
    """
    if not frame: return

    if recv_fmt == RecvFmt.bgr:
        gl_format = GL_BGRA
    else:
        gl_format = GL_RGBA

    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_w, tex_h, 0,
                 gl_format, GL_UNSIGNED_BYTE, frame)
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    src_ratio = tex_w / tex_h
    dst_ratio = win_w / win_h

    if src_ratio > dst_ratio:
        scale_x = 1.0
        scale_y = dst_ratio / src_ratio
    else:
        scale_x = src_ratio / dst_ratio
        scale_y = 1.0

    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0); glVertex2f(-scale_x, -scale_y)
    glTexCoord2f(1.0, 1.0); glVertex2f( scale_x, -scale_y)
    glTexCoord2f(1.0, 0.0); glVertex2f( scale_x,  scale_y)
    glTexCoord2f(0.0, 0.0); glVertex2f(-scale_x,  scale_y)
    glEnd()

def render_waiting_message():
    """接続待機中に表示する画面を描画する
    """
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

def init_window(title: str, width: int, height: int, fullscreen: bool):
    """SDL2とOpenGLを使用してウィンドウを初期化する
    """
    if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
        raise RuntimeError(f"SDL_Initエラー: {sdl2.SDL_GetError()}")

    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 2)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 1)
    
    flags = sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_RESIZABLE
    if fullscreen:
        flags |= sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP

    window = sdl2.SDL_CreateWindow(
        title.encode('utf-8'),
        sdl2.SDL_WINDOWPOS_CENTERED, sdl2.SDL_WINDOWPOS_CENTERED,
        width, height,
        flags
    )
    if not window:
        raise RuntimeError(f"SDL_CreateWindowエラー: {sdl2.SDL_GetError()}")

    sdl2.SDL_GL_CreateContext(window)
    sdl2.SDL_ShowCursor(sdl2.SDL_DISABLE)
    return window

def ndi_capture_thread(receiver: Receiver, vf: VideoFrameSync, frame_queue: queue.Queue, stop_event: threading.Event):
    """バックグラウンドでNDIフレームをキャプチャし続けるワーカースレッド
    """
    while not stop_event.is_set():
        if not receiver.is_connected():
            break
        
        receiver.frame_sync.capture_video()
        tex_w, tex_h = vf.get_resolution()
        
        if tex_w > 0 and tex_h > 0 and vf.get_data_size() > 0:
            try:
                # キューが満杯の場合は古いフレームを捨てて新しいフレームを入れる
                frame_queue.put_nowait((bytes(vf), tex_w, tex_h))
            except queue.Full:
                try:
                    frame_queue.get_nowait() # 古いものを捨てる
                    frame_queue.put_nowait((bytes(vf), tex_w, tex_h)) # 新しいものを入れる
                except queue.Empty:
                    pass # タイミングの問題で空になった場合は何もしない
        else:
            time.sleep(0.001) # CPUの過剰な使用を防ぐ
    
    frame_queue.put(None) # スレッド終了のシグナル
    click.echo("キャプチャスレッドを停止しました。")

def play_sdl(options: Options):
    """SDLを使用してNDIストリームを再生するメインロジック。
    """
    window = init_window("NDI Viewer", 1280, 720, options.fullscreen)
    
    w_ptr, h_ptr = sdl2.c_int(), sdl2.c_int()
    sdl2.SDL_GetWindowSize(window, w_ptr, h_ptr)
    win_w, win_h = w_ptr.value, h_ptr.value

    glEnable(GL_TEXTURE_2D)
    glViewport(0, 0, win_w, win_h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    texture_id = glGenTextures(1)
    
    finder = None
    receiver = None
    capture_thread = None
    stop_thread_event = None
    
    running = True
    event = sdl2.SDL_Event()
    
    is_connected = False
    reconnect_cooldown_until = 0
    
    last_frame_data = None
    last_frame_w, last_frame_h = 0, 0

    try:
        finder = Finder()
        vf = VideoFrameSync()
        frame_queue = queue.Queue(maxsize=2)

        while running:
            while sdl2.SDL_PollEvent(event):
                if event.type == sdl2.SDL_QUIT:
                    running = False
                elif event.type == sdl2.SDL_KEYDOWN and event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                    running = False
                elif event.type == sdl2.SDL_WINDOWEVENT and event.window.event == sdl2.SDL_WINDOWEVENT_RESIZED:
                    win_w, win_h = event.window.data1, event.window.data2
                    glViewport(0, 0, win_w, win_h)

            if not is_connected:
                # --- 切断状態の処理 ---
                if time.time() < reconnect_cooldown_until:
                    render_waiting_message()
                else:
                    click.echo("NDI接続を試行しています...")
                    try:
                        source = get_source(finder, options.sender_name)
                        receiver = Receiver(
                            color_format=options.recv_fmt.value,
                            bandwidth=options.recv_bandwidth.value,
                        )
                        receiver.frame_sync.set_video_frame(vf)
                        receiver.set_source(source)

                        i = 0
                        while not receiver.is_connected():
                            if i > 30: raise Exception('タイムアウト')
                            time.sleep(0.1)
                            i += 1
                        
                        wait_for_first_frame(receiver)
                        
                        stop_thread_event = threading.Event()
                        capture_thread = threading.Thread(
                            target=ndi_capture_thread,
                            args=(receiver, vf, frame_queue, stop_thread_event)
                        )
                        capture_thread.start()
                        is_connected = True
                        click.echo("NDIソースに接続しました。")

                    except Exception as e:
                        click.echo(f"接続試行中にエラー: {e}", err=True)
                        if receiver:
                            receiver = None; gc.collect()
                        reconnect_cooldown_until = time.time() + 5.0 # 5秒後に再試行
                
                render_waiting_message()

            else:
                # --- 接続状態の処理 ---
                try:
                    frame_info = frame_queue.get_nowait()
                    if frame_info is None: # スレッド終了シグナル
                        raise ConnectionError("キャプチャスレッドが停止しました。")
                    
                    last_frame_data, last_frame_w, last_frame_h = frame_info
                    render_texture(last_frame_data, last_frame_w, last_frame_h, win_w, win_h, texture_id, options.recv_fmt)

                except queue.Empty:
                    # 新しいフレームがない場合は、最後のフレームを再描画
                    render_texture(last_frame_data, last_frame_w, last_frame_h, win_w, win_h, texture_id, options.recv_fmt)
                except (ConnectionError, BrokenPipeError) as e:
                    click.echo(f"接続が失われました: {e}", err=True)
                    is_connected = False
                    if stop_thread_event: stop_thread_event.set()
                    if capture_thread: capture_thread.join()
                    capture_thread = None
                    if receiver: receiver = None; gc.collect()
                    # キューをクリア
                    while not frame_queue.empty(): frame_queue.get()
                    reconnect_cooldown_until = time.time() + 5.0

            sdl2.SDL_GL_SwapWindow(window)
            time.sleep(0.001) # メインループのCPU使用率を抑制

    finally:
        click.echo("クリーンアップ処理を実行しています...")
        if stop_thread_event: stop_thread_event.set()
        if capture_thread: capture_thread.join()
        if finder and hasattr(finder, 'destroy'): finder.destroy()
        
        glDeleteTextures(1, [texture_id])
        sdl2.SDL_DestroyWindow(window)
        sdl2.SDL_Quit()
        click.echo("プログラムを終了しました。")


@click.command()
@click.option(
    '-s', '--sender-name', type=str, default='ffmpeg_sender', show_default=True, help='接続するNDIソース名')
@click.option(
    '-f', '--recv-fmt', type=click.Choice(choices=[m.name for m in RecvFmt]), default='rgb', show_default=True, help='受信するピクセルフォーマット')
@click.option(
    '-b', '--recv-bandwidth', type=click.Choice(choices=[m.name for m in Bandwidth]), default='highest', show_default=True, help='受信帯域')
@click.option(
    '--fullscreen', is_flag=True, help='フルスクリーンモードで起動する')
def main(sender_name: str, recv_fmt: str, recv_bandwidth: str, fullscreen: bool):
    """SDL2とPyOpenGLを使用してNDIストリームを表示するビューア"""
    options = Options(
        sender_name=sender_name,
        recv_fmt=RecvFmt.from_str(recv_fmt),
        recv_bandwidth=Bandwidth.from_str(recv_bandwidth),
        fullscreen=fullscreen,
    )
    try:
        play_sdl(options)
    except Exception as e:
        click.echo(f"致命的なエラーが発生しました: {e}", err=True)


if __name__ == '__main__':
    main()
