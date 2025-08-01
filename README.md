# utone-ndi-utils
NDIを活用することでDJイベント "The Utopia Tone" の映像伝送をIPネットワーク上に移行します。  
PythonとSDL2を使用してNDIソースを受信し、全画面表示を行ったり、OpenCVを使用してNDIソースの送信を行います。

## 使い方
動作にはPython3及びavahi-daemon、libgl1-mesa-devが必要です。
以下のコマンドで必要なパッケージをインストールしてください。
```bash
$ sudo apt install python3 python3-pip python3-venv avahi-daemon libgl1-mesa-dev
```

次に、リポジトリをクローンし、仮想環境を作成して依存関係をインストールします。
```bash
$ git clone https://github.com/TechnoTUT/utone-ndi-utils.git
$ cd utone-ndi-utils
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

NDIソースを受信して全画面表示するには、以下のコマンドを実行します。
```bash
$ python3 rx_sdl2.py -s "<NDI Source Name>" --fullscreen
```
`<NDI Source Name>`は、受信したいNDIソースの名前に置き換えてください。

NDIソースを送信するには、以下のコマンドを実行します。
```bash
$ python3 tx.py
```

## 自動起動設定
Systemdを使用して自動起動する場合は、以下の手順を実行します。なお、GUIなしの環境でも動作します。  
ExecuteStartとWorkingDirectoryのパスを適切なものに変更します。  
以下では、例として受信側の設定を示しますが、送信側も同様の手順で設定できます。送信側の設定を行う場合は、`rx`を`tx`に置き換えてください。
```bash
$ mkdir -p ~/.config/systemd/user
$ cp systemd-example/ndi-rx.service ~/.config/systemd/user/
$ vim ~/.config/systemd/user/ndi-rx.service
```

自動起動を有効にします。
```bash
$ systemctl --user daemon-reload
$ systemctl --user enable --now ndi-rx.service
```

ログイン時でなく、システム起動時に自動起動する場合は、lingerを有効にします。
```bash
$ sudo loginctl enable-linger username
```