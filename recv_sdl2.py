from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING
from typing_extensions import Self
import enum
import time

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
    lowest = RecvBandwidth.lowest      #: 最低
    highest = RecvBandwidth.highest    #: 最高

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
    """最初の数フレームはデータを含まないことがあるため、
    データを含む最初のフレームを受信するまで待機する
    """
    vf = receiver.frame_sync.video_frame
    assert vf is not None
    frame_rate = vf.get_frame_rate()
    # フレームレートが0の場合のフォールバック
    wait_time = float(1 / frame_rate) if frame_rate > 0 else 0.033
    click.echo('最初のフレームを待機中...')
    while receiver.is_connected():
        receiver.frame_sync.capture_video()
        resolution = vf.get_resolution()
        if min(resolution) > 0 and vf.get_data_size() > 0:
            click.echo('フレームを取得しました。')
            return
        time.sleep(wait_time)

def render_texture(frame: bytes, tex_w: int, tex_h: int, win_w: int, win_h: int, texture_id: int, recv_fmt: RecvFmt):
    """受信したフレームデータをOpenGLテクスチャとしてアスペクト比を維持して描画する
    """
    # 受信フォーマットに応じてOpenGLのピクセルフォーマットを決定
    if recv_fmt == RecvFmt.bgr:
        gl_format = GL_BGRA
    else:  # UYVY (cyndilibによりRGBAに変換) および RGB の場合
        gl_format = GL_RGBA

    glBindTexture(GL_TEXTURE_2D, texture_id)
    # テクスチャデータをGPUに転送
    # 内部フォーマットは互換性のためGL_RGBAに固定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_w, tex_h, 0,
                 gl_format, GL_UNSIGNED_BYTE, frame)
    
    # テクスチャのフィルタリング設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

    # 画面をクリア
    glClear(GL_COLOR_BUFFER_BIT)

    # アスペクト比を計算して維持
    src_ratio = tex_w / tex_h
    dst_ratio = win_w / win_h

    if src_ratio > dst_ratio:
        scale_x = 1.0
        scale_y = dst_ratio / src_ratio
    else:
        scale_x = src_ratio / dst_ratio
        scale_y = 1.0

    # 画面に四角形を描画し、テクスチャをマッピング
    # Y座標を反転させて上下反転を修正する
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0); glVertex2f(-scale_x, -scale_y)
    glTexCoord2f(1.0, 1.0); glVertex2f( scale_x, -scale_y)
    glTexCoord2f(1.0, 0.0); glVertex2f( scale_x,  scale_y)
    glTexCoord2f(0.0, 0.0); glVertex2f(-scale_x,  scale_y)
    glEnd()

def init_window(title: str, width: int, height: int, fullscreen: bool):
    """SDL2とOpenGLを使用してウィンドウを初期化する
    """
    if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
        raise RuntimeError(f"SDL_Initエラー: {sdl2.SDL_GetError()}")

    # OpenGLのバージョンを指定
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

def run_display_loop(frame_gen, options: Options):
    """メインの表示ループ。フレームを継続的に取得し、画面に描画する
    """
    # 最初のフレームを取得して解像度を得る
    try:
        frame, tex_w, tex_h = next(frame_gen)
    except StopIteration:
        click.echo("フレームジェネレータからフレームを取得できませんでした。終了します。")
        return
    
    # 初期ウィンドウサイズはフレーム解像度に合わせる
    win_w, win_h = tex_w, tex_h
    
    window = init_window("NDI Viewer", win_w, win_h, options.fullscreen)
    
    # ウィンドウ作成後、実際のウィンドウサイズを取得し直す
    w_ptr = sdl2.c_int()
    h_ptr = sdl2.c_int()
    sdl2.SDL_GetWindowSize(window, w_ptr, h_ptr)
    win_w, win_h = w_ptr.value, h_ptr.value

    # OpenGL初期化
    glEnable(GL_TEXTURE_2D)
    glViewport(0, 0, win_w, win_h) # 取得した正しいサイズでビューポートを設定
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    texture_id = glGenTextures(1)

    event = sdl2.SDL_Event()
    running = True

    while running:
        # イベント処理
        while sdl2.SDL_PollEvent(event):
            if event.type == sdl2.SDL_QUIT:
                running = False
            elif event.type == sdl2.SDL_WINDOWEVENT:
                if event.window.event == sdl2.SDL_WINDOWEVENT_RESIZED:
                    # ウィンドウサイズ変更時にビューポートを更新
                    win_w, win_h = event.window.data1, event.window.data2
                    glViewport(0, 0, win_w, win_h)

        # アスペクト比を維持して描画
        render_texture(frame, tex_w, tex_h, win_w, win_h, texture_id, options.recv_fmt)

        # ダブルバッファリングをスワップして画面を更新
        sdl2.SDL_GL_SwapWindow(window)

        # 次のフレームを取得
        try:
            frame, tex_w, tex_h = next(frame_gen)
        except StopIteration:
            break


    # クリーンアップ
    glDeleteTextures(1, [texture_id])
    sdl2.SDL_DestroyWindow(window)
    sdl2.SDL_Quit()

def ndi_frame_generator(receiver, vf):
    """NDIフレームを継続的に生成するジェネレータ
    """
    while receiver.is_connected():
        receiver.frame_sync.capture_video()
        xres, yres = vf.get_resolution()
        if xres > 0 and yres > 0:
            yield bytes(vf), xres, yres

def play_sdl(options: Options):
    """SDLを使用してNDIストリームを再生するメインロジック
    """
    with Finder() as finder:
        source = get_source(finder, options.sender_name)
        
        # Receiverはコンテキストマネージャをサポートしていないため、with文を使用しない
        receiver = Receiver(
            color_format=options.recv_fmt.value,
            bandwidth=options.recv_bandwidth.value,
        )
        
        vf = VideoFrameSync()
        receiver.frame_sync.set_video_frame(vf)
        receiver.set_source(source)

        i = 0
        while not receiver.is_connected():
            if i > 30:
                raise Exception('タイムアウト：NDIソースへの接続に失敗しました。')
            time.sleep(0.1)
            i += 1

        wait_for_first_frame(receiver)

        # ストリームが安定するまで、最初の数フレームを意図的に破棄する
        # これにより、初期の不完全なフレームによる問題を回避する
        click.echo('ストリームを安定させています...')
        frame_rate = vf.get_frame_rate()
        wait_time = float(1 / frame_rate) if frame_rate > 0 else 0.033
        for _ in range(5): # 5フレームほど破棄してみる
            receiver.frame_sync.capture_video()
            time.sleep(wait_time)

        frame_gen = ndi_frame_generator(receiver, vf)
        run_display_loop(frame_gen, options)

@click.command()
@click.option(
    '-s', '--sender-name',
    type=str,
    default='ffmpeg_sender',
    show_default=True,
    help='接続するNDIソース名',
)
@click.option(
    '-f', '--recv-fmt',
    type=click.Choice(choices=[m.name for m in RecvFmt]),
    default='rgb',
    show_default=True,
    show_choices=True,
    help='受信するピクセルフォーマット'
)
@click.option(
    '-b', '--recv-bandwidth',
    type=click.Choice(choices=[m.name for m in Bandwidth]),
    default='highest',
    show_default=True,
    show_choices=True,
    help='受信帯域'
)
@click.option(
    '--fullscreen',
    is_flag=True,
    help='フルスクリーンモードで起動する'
)
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
        click.echo(f"エラーが発生しました: {e}", err=True)


if __name__ == '__main__':
    main()
