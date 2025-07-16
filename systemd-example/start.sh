#!/bin/bash
# venv activation
source /path/to/ndi-receiver/venv/bin/activate
exec python3 /path/to/ndi-receiver/recv_sdl2.py -s "__SOURCE_NAME__" --fullscreen
