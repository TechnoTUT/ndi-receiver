# ndi-receiver
PythonとSDL2を使用してNDIソースを受信し、全画面表示します。  
NDIを活用することでDJイベント "The Utopia Tone" の映像伝送をIPネットワーク上に移行します。

## 使い方
動作にはPython3及びavahi-daemon、libgl1-mesa-devが必要です。
以下のコマンドで必要なパッケージをインストールしてください。
```bash
$ sudo apt install python3 python3-pip python3-venv avahi-daemon libgl1-mesa
```

次に、リポジトリをクローンし、仮想環境を作成して依存関係をインストールします。
```bash
$ git clone https://github.com/TechnoTUT/ndi-receiver.git
$ cd ndi-receiver
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ python recv_sdl2.py -s "<NDI Source Name>" --fullscreen
```

Systemdを使用して自動起動する場合は、以下の手順を実行します。なお、GUIなしの環境でも動作します。
ExecuteStartとWorkingDirectoryのパスを適切なものに変更します。
```bash
$ mkdir -p ~/.config/systemd/user
$ cp systemd-example/ndi-receiver.service ~/.config/systemd/user/
$ vim ~/.config/systemd/user/ndi-receiver.service
```

自動起動を有効にします。
```bash
$ systemctl --user daemon-reload
$ systemctl --user enable --now ndi-receiver.service
```

ログイン時でなく、システム起動時に自動起動する場合は、lingerを有効にします。
```bash
$ sudo loginctl enable-linger username
```