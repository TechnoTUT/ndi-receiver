# ndi-receiver
PythonとSDL2を使用してNDIソースを受信し、全画面表示します。  
NDIを活用することでDJイベント "The Utopia Tone" の映像伝送をIPネットワーク上に移行します。
## 使い方
```
$ git clone https://github.com/TechnoTUT/ndi-receiver.git
$ cd ndi-receiver
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ python recv_sdl2.py -s "<NDI Source Name>" --fullscreen
```