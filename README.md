# naturetech_bird_demo

NATURE TECH! 2026 向けの鳥の声AI判定・鳥の鳴きまねチャレンジ用デモアプリ

## 機能

- BirdNET による鳥の声AI判定
- 鳥の鳴きまねチャレンジ

## 画面構成

| URL | 用途 |
|---|---|
| `http://localhost:8000/index.html` | 入口画面 |
| `http://localhost:8000/birdnet` | 鳥の声AI判定画面 |
| `http://localhost:8000/mimic` | 鳥の鳴きまねチャレンジ画面 |
| `http://<ホストのLAN IP>:8000/control` | 外部デバイス用操作パネル |

モニター側のブラウザでは、`localhost` の URL を開いてください  
外部デバイス用の操作パネルは、スマートフォン、タブレット、別 PC などから LAN 側 URL を開きます

## 起動方法

```bash
python app.py
```

## 環境構築

## 1. Homebrew / Python の確認
```bash
which brew
brew --version

which python3
python3 --version
```

#### Homebrew がインストールされていない場合は、以下でインストール
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
#### Apple Silicon Mac で brew にパスが通っていない場合は、以下を実行
```bash
eval "$(/opt/homebrew/bin/brew shellenv)"
```

## 2. 必要なシステムライブラリをインストール
```bash
brew install python@3.11 portaudio libsndfile git
```

#### GitHub CLI を使う場合は、gh もインストール
```bash
brew install gh
```

## 3. GitHub の認証
```bash
gh auth login
```
質問への回答は以下
| 質問 | 回答 |
|---|---|
| What account do you want to log into? | GitHub.com |
| What is your preferred protocol for Git operations? | HTTPS |
| Authenticate Git with your GitHub credentials? | Yes |
| How would you like to authenticate GitHub CLI? | Login with a web browser |

## 4. リポジトリの clone
```bash
git clone https://github.com/tfujita-aist/naturetech_bird_demo
cd naturetech_bird_demo
```
または
```bash
gh repo clone tfujita-aist/naturetech_bird_demo
cd naturetech_bird_demo
```
すでに clone 済みで、最新版を取得する場合は、
```bash
cd naturetech_bird_demo
git pull --ff-only origin main
```

## 5. Python 仮想環境の作成
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```
仮想環境を有効化するには
```bash
source .venv/bin/activate
```

## 6. Python パッケージのインストール
```bash
pip install -r requirements.txt
```

## 7. BirdNET モデルファイルの配置
```bash
mkdir -p models/BirdNET_V2.4
curl -L "https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite.zip?download=1" -o models/BirdNET_V2.4/BirdNET_v2.4_tflite.zip
unzip -o models/BirdNET_V2.4/BirdNET_v2.4_tflite.zip -d models/BirdNET_V2.4
```

## 8. 音声デバイスの確認
```bash
python tools/list_audio_devices.py
```
YAMAHA UR22C などの 2ch オーディオインタフェースを使う場合は、config.yaml の audio を以下の様に設定
```YAML
audio:
  sample_rate: 48000
  block_size: 1024
  channels: 2
  device_index: < UR22C の番号>
  device_name_contains: "Steinberg UR22C"
  birdnet_channel: 0
  mimic_channel: 1
```
MacBook 内蔵マイクを使う場合は、config.yaml の audio を以下の様に設定
```YAML
audio:
  sample_rate: 48000
  block_size: 1024
  channels: 1
  device_index: < MacBook の番号>
  device_name_contains: "MacBook Proのマイク" or "MacBook Airのマイク" など
  birdnet_channel: 0
  mimic_channel: 0
```

## 9. アプリケーションの起動
```bash
python app.py
```
起動すると、ターミナルにアクセス用の URL が表示される

## 10. マイクの権限
/mimic 画面では、最初に「音声を有効化」を押下する  

録音中のリアルタイム波形を表示するには、ブラウザで /mimic に対するマイク使用を許可する  

Mac 側でもマイクの権限が必要
```
システム設定 > プライバシーとセキュリティ > マイク > {ブラウザのアプリ} を ON
```

## 11. ２回目以降の起動
```bash
cd naturetech_bird_demo
source .venv/bin/activate
python app.py
```
