# ComfyUI-GPT_SoVITS
a comfyui custom node for [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)! you can voice cloning and tts in comfyui now
<div>
  <figure>
  <img alt='webpage' src="web.png?raw=true" width="600px"/>
  <figure>
</div>

## Features
- `srt` file for subtitle was supported
- mutiple speaker was supported in finetune and inference by `srt`
- huge comfyui custom nodes can merge in gpt_sovits

## How to use
make sure `ffmpeg` is worked in your commandline
for Linux
```
apt update
apt install ffmpeg
```
for Windows,you can install `ffmpeg` by [WingetUI](https://github.com/marticliment/WingetUI) automatically

then!
```
git clone https://github.com/AIFSH/ComfyUI-GPT_SoVITS.git
cd ComfyUI-GPT_SoVITS
pip install -r requirements.txt
```
`weights` will be downloaded from huggingface automatically! if you in china,make sure your internet attach the huggingface
or if you still struggle with huggingface, you may try follow [hf-mirror](https://hf-mirror.com/) to config your env.

或者下载[权重文件](https://pan.quark.cn/s/e5057be01087)解压后把`pretrained_models`整个文件夹放进`ComfyUI-GPT_SoVITS`目录

## Windows
There is a portable standalone build for Windows that should work for running on Nvidia GPUs and cuda>=11.8,
click [the link](https://www.bilibili.com/video/BV1qx4y1h7T2) to download

## Tutorial
- [Demo](https://www.bilibili.com/video/BV1yC411G7NJ)
- [Demo for mutiple speaker](https://www.bilibili.com/video/BV1QC41137Wq/)
- [FULL WorkFLOW](https://www.bilibili.com/video/BV1pp421D7qa)
## My other nodes you may need
- [ComfyUI-UVR5](https://github.com/AIFSH/ComfyUI-UVR5)
- [ComfyUI-IP_LAP](https://github.com/AIFSH/ComfyUI-IP_LAP)

## WeChat Group && Donate
<div>
  <figure>
  <img alt='Wechat' src="wechat.jpg?raw=true" width="300px"/>
  <img alt='donate' src="donate.jpg?raw=true" width="300px"/>
  <figure>
</div>

## Thanks
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
