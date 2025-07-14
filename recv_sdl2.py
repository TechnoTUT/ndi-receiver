from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING
from typing_extensions import Self
import enum
import time
import subprocess
import shlex

import click

from cyndilib.wrapper.ndi_structs import FourCC
from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
from cyndilib.video_frame import VideoFrameSync
from cyndilib.receiver import Receiver
from cyndilib.finder import Finder
if TYPE_CHECKING:
    from cyndilib.finder import Source

import sdl2
import sdl2.ext
from OpenGL.GL import *
from OpenGL.GLU import *

pix_fmts = {
    FourCC.UYVY: 'uyvy422',
    FourCC.NV12: 'nv12',
    FourCC.RGBA: 'rgba',
    FourCC.BGRA: 'bgra',
    FourCC.RGBX: 'rgba',
    FourCC.BGRX: 'bgra',
}
"""Mapping of :class:`FourCC <cyndilib.wrapper.ndi_structs.FourCC>` types to
ffmpeg's ``pix_fmt`` definitions
"""


class RecvFmt(enum.Enum):
    """Pixel format to receive (mapped to values of
    :class:`cyndilib.wrapper.ndi_recv.RecvColorFormat`)
    """
    uyvy = RecvColorFormat.UYVY_RGBA    #: UYVY (RGBA if alpha is present)
    rgb = RecvColorFormat.RGBX_RGBA     #: RGB / RGBA
    bgr = RecvColorFormat.BGRX_BGRA     #: BGR / BGRA

    @classmethod
    def from_str(cls, name: str) -> Self:
        return cls.__members__[name]


class Bandwidth(enum.Enum):
    """Receive bandwidth
    """
    lowest = RecvBandwidth.lowest      #: Lowest
    highest = RecvBandwidth.highest    #: Highest

    @classmethod
    def from_str(cls, name: str) -> Self:
        return cls.__members__[name]


class Options(NamedTuple):
    """Options set through the cli
    """
    sender_name: str = 'ffmpeg_sender'
    """The name of the |NDI| source to connect to"""

    recv_fmt: RecvFmt = RecvFmt.uyvy
    """Receive pixel format"""

    recv_bandwidth: Bandwidth = Bandwidth.highest
    """Receive bandwidth"""

    ffplay: str = 'ffplay'
    """Name/Path of the ``ffplay`` executable"""


def get_source(finder: Finder, name: str) -> Source:
    """Use the Finder to search for an NDI source by name using either its
    full name or its :attr:`~cyndilib.finder.Source.stream_name`
    """
    click.echo('waiting for ndi sources...')
    finder.wait_for_sources(10)
    for source in finder:
        if source.name == name or source.stream_name == name:
            return source
    raise Exception(f'source not found. {finder.get_source_names()=}')


def wait_for_first_frame(receiver: Receiver) -> None:
    """The first few frames contain no data. Capture frames until the first
    non-empty one
    """
    vf = receiver.frame_sync.video_frame
    assert vf is not None
    frame_rate = vf.get_frame_rate()
    wait_time = float(1 / frame_rate)
    click.echo('waiting for frame...')
    while receiver.is_connected():
        receiver.frame_sync.capture_video()
        resolution = vf.get_resolution()
        if min(resolution) > 0 and vf.get_data_size() > 0:
            click.echo('have frame')
            return
        time.sleep(wait_time)

def init_window(title: str, width: int, height: int):
    if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
        raise RuntimeError("SDL_Init Error")

    window = sdl2.SDL_CreateWindow(
        title.encode('utf-8'),
        sdl2.SDL_WINDOWPOS_CENTERED, sdl2.SDL_WINDOWPOS_CENTERED,
        width, height,
        sdl2.SDL_WINDOW_OPENGL
    )
    if not window:
        raise RuntimeError("SDL_CreateWindow Error")

    sdl2.SDL_GL_CreateContext(window)
    return window

def draw_frame(data: bytes, width: int, height: int, fourcc: str):
    # 簡略化: RGBA 前提
    glClear(GL_COLOR_BUFFER_BIT)
    glRasterPos2f(-1, -1)
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glFlush()

def run_display_loop(frame_gen, width: int, height: int):
    window = init_window("NDI Viewer", width, height)
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, width, 0, height)

    glPixelZoom(1.0, -1.0)
    glRasterPos2i(0, height)

    event = sdl2.SDL_Event()
    running = True
    while running:
        while sdl2.SDL_PollEvent(event):
            if event.type == sdl2.SDL_QUIT:
                running = False

        frame = next(frame_gen, None)
        if frame is not None:
            glClear(GL_COLOR_BUFFER_BIT)
            glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, frame)
            glFlush()

        sdl2.SDL_GL_SwapWindow(window)

    sdl2.SDL_DestroyWindow(window)
    sdl2.SDL_Quit()



def ndi_frame_generator(receiver, vf):
    while receiver.is_connected():
        receiver.frame_sync.capture_video()
        yield bytes(vf)

def play_sdl(options: Options):
    with Finder() as finder:
        source = get_source(finder, options.sender_name)
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
                raise Exception('timeout')
            time.sleep(0.1)
            i += 1

        wait_for_first_frame(receiver)
        xres, yres = vf.get_resolution()

        frame_gen = ndi_frame_generator(receiver, vf)
        run_display_loop(frame_gen, xres, yres)

@click.command()
@click.option(
    '-s', '--sender-name',
    type=str,
    default='ffmpeg_sender',
    show_default=True,
    help='The NDI source name to connect to',
)
@click.option(
    '-f', '--recv-fmt',
    type=click.Choice(choices=[m.name for m in RecvFmt]),
    default='uyvy',
    show_default=True,
    show_choices=True,
    help='Pixel format'
)
@click.option(
    '-b', '--recv-bandwidth',
    type=click.Choice(choices=[m.name for m in Bandwidth]),
    default='highest',
    show_default=True,
    show_choices=True,
)
@click.option(
    '--ffplay',
    type=str,
    default='ffplay',
    show_default=True,
    help='Name/Path of the "ffplay" executable',
)
def main(sender_name: str, recv_fmt: str, recv_bandwidth: str, ffplay: str):
    options = Options(
        sender_name=sender_name,
        recv_fmt=RecvFmt.rgb, #RecvFmt.from_str(recv_fmt),
        recv_bandwidth=Bandwidth.from_str(recv_bandwidth),
        ffplay='',
    )
    play_sdl(options)


if __name__ == '__main__':
    main()
