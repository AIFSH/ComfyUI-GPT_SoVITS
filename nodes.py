import os
import re
import audiotsm
import audiotsm.io.wav
from time import time as ttime
import folder_paths
from pydub import AudioSegment
from tools.i18n.i18n import I18nAuto
from srt import parse as SrtPare
from .inference import dict_language,get_tts_wav
from .finetune import open1abc,default_batch_size,open1Ba,open1Bb

i18n = I18nAuto()

parent_directory = os.path.dirname(os.path.abspath(__file__))
input_path = folder_paths.get_input_directory()
out_path = folder_paths.get_output_directory()
language_list = [i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")]
weights_path = os.path.join(parent_directory,"pretrained_models")
SoVITS_weight_root = os.path.join(out_path,"sovits_weights")
os.makedirs(SoVITS_weight_root,exist_ok=True)

GPT_weight_root = os.path.join(out_path,"gpt_weights")
os.makedirs(GPT_weight_root,exist_ok=True)
sovits_files = sorted(os.listdir(SoVITS_weight_root),reverse=True)
        
gpt_files = sorted(os.listdir(GPT_weight_root),reverse=True)

class GPT_SOVITS_TTS:
    @classmethod
    def INPUT_TYPES(s):
        how_to_cuts = [i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"), i18n("按中文句号。切"), i18n("按英文句号.切"), i18n("按标点符号切"), ]
        
        return {"required":
                    {
                     "renfer_audio":("AUDIO",),
                     "refer_srt":("SRT",),
                     "refer_language":(language_list,{
                         "default": i18n("中文")
                     }),
                     "text": ("STRING",{
                         "default": "你好啊！世界",
                         "multiline": True
                     }),
                     "text_language":(language_list,{
                         "default": i18n("中文")
                     }),
                     "gpt_weight":(gpt_files,),
                     "sovits_weight":(sovits_files,),
                     "how_to_cut":(how_to_cuts,{
                         "default": i18n("凑四句一切")
                     }),
                     "top_k":("INT",{
                         "default":20,
                         "min":1,
                         "max": 100,
                         "step": 1,
                         "display": "slider"
                     }),
                     "top_p":("FLOAT",{
                         "default":1,
                         "min":0,
                         "max": 1,
                         "step": 0.05,
                         "display": "slider"
                     }),
                     "temperature":("FLOAT",{
                         "default":1,
                         "min":0,
                         "max": 1,
                         "step": 0.05,
                         "display": "slider"
                     }),
                    }
                }
    CATEGORY = "AIFSH_GPT_SOVITS"
    RETURN_TYPES = ('AUDIO',)
    OUTPUT_NODE = False

    FUNCTION = "get_tts_wav"

    def get_tts_wav(self,renfer_audio,refer_srt,refer_language,
            text,text_language,gpt_weight,sovits_weight,
            how_to_cut,top_k,top_p,temperature):
        
        with open(refer_srt, 'r', encoding="utf-8") as file:
            file_content = file.read()
        prompt_language = dict_language[refer_language]
        dot_ = "。" if 'zh' in prompt_language else '.'
        prompt_text = f'{dot_}'.join([sub.content for sub in list(SrtPare(file_content))])
        print(f"prompt_text:{prompt_text}")
        outfile = os.path.join(out_path, f"{ttime()}_gpt_sovits_tts.wav")
        gpt_weight = os.path.join(GPT_weight_root, gpt_weight)
        sovits_weight = os.path.join(SoVITS_weight_root, sovits_weight)
        get_tts_wav(renfer_audio,prompt_text,prompt_language,
            text,text_language,how_to_cut,top_k,top_p,temperature,
            gpt_weight,sovits_weight,outfile)
        
        return (outfile,)
    
class GPT_SOVITS_INFER:
    @classmethod
    def INPUT_TYPES(s):
        how_to_cuts = [i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"), i18n("按中文句号。切"), i18n("按英文句号.切"), i18n("按标点符号切"), ]
        
        return {"required":
                    {
                     "renfer_audio":("AUDIO",),
                     "refer_srt":("SRT",),
                     "if_aliginment":("BOOLEAN",{
                         "default": False
                     }),
                     "if_mutiple_speaker":("BOOLEAN",{
                         "default": False
                     }),
                     "refer_language":(language_list,{
                         "default": i18n("中文")
                     }),
                     "text_srt":("SRT",),
                     "text_language":(language_list,{
                         "default": i18n("中文")
                     }),
                     "gpt_weight":(gpt_files,),
                     "sovits_weight":(sovits_files,),
                     "how_to_cut":(how_to_cuts,{
                         "default": i18n("不切")
                     }),
                     "top_k":("INT",{
                         "default":20,
                         "min":1,
                         "max": 100,
                         "step": 1,
                         "display": "slider"
                     }),
                     "top_p":("FLOAT",{
                         "default":1,
                         "min":0,
                         "max": 1,
                         "step": 0.05,
                         "display": "slider"
                     }),
                     "temperature":("FLOAT",{
                         "default":1,
                         "min":0,
                         "max": 1,
                         "step": 0.05,
                         "display": "slider"
                     }),
                    }
                }
    CATEGORY = "AIFSH_GPT_SOVITS"
    RETURN_TYPES = ('AUDIO',)
    OUTPUT_NODE = False

    FUNCTION = "get_tts_wav"

    def get_tts_wav(self,renfer_audio,refer_srt,if_aliginment,
                    if_mutiple_speaker,refer_language,text_srt,text_language,
                    gpt_weight,sovits_weight,how_to_cut,top_k,top_p,temperature):
        
        prompt_language = dict_language[refer_language]

        refer_srt_path = folder_paths.get_annotated_filepath(refer_srt)
        text_srt_path = folder_paths.get_annotated_filepath(text_srt)
        with open(refer_srt_path, 'r', encoding="utf-8") as file:
            refer_file_content = file.read()
        with open(text_srt_path, 'r', encoding="utf-8") as file:
            text_file_content = file.read()
        
        refer_wav_root = os.path.join(input_path, "gpt_sovits_infer")
        os.makedirs(refer_wav_root,exist_ok=True)
        audio_path = folder_paths.get_annotated_filepath(renfer_audio)
        audio_seg = AudioSegment.from_file(audio_path)
        new_audio_seg = AudioSegment.silent(0)
        refer_subtitles = list(SrtPare(refer_file_content))
        for i, (refer_sub, text_sub) in enumerate(zip(refer_subtitles, list(SrtPare(text_file_content)))):
            start_time = refer_sub.start.total_seconds() * 1000
            end_time = refer_sub.end.total_seconds() * 1000
            if i == 0:
                new_audio_seg += audio_seg[:start_time]
            refer_wav_seg = audio_seg[start_time:end_time]
            refer_wav = os.path.join(refer_wav_root, f"{i}_gpt_sovits_refer.wav")
            refer_wav_seg.export(refer_wav, format='wav')

            outfile = os.path.join(refer_wav_root, f"{i}_gpt_sovits_infer.wav")
            text = text_sub.content
            refer_text = refer_sub.content
            if if_mutiple_speaker:
                speaker_name = f"speaker_{text[0]}"
                text = text[1:]
                refer_text = refer_text[1:]
                gpt_weight = sorted([f for f in os.listdir(GPT_weight_root) if speaker_name in f])[0]
                gpt_weight = os.path.join(GPT_weight_root, gpt_weight)
                sovits_weight = sorted([f for f in os.listdir(SoVITS_weight_root) if speaker_name in f])[0]
                sovits_weight = os.path.join(SoVITS_weight_root, sovits_weight)
            else:
                gpt_weight = os.path.join(GPT_weight_root, gpt_weight)
                sovits_weight = os.path.join(SoVITS_weight_root, sovits_weight)
                
            get_tts_wav(refer_wav,refer_text,prompt_language,
                text,text_language,how_to_cut,top_k,top_p,temperature,
                gpt_weight,sovits_weight,outfile)
            
            text_audio = AudioSegment.from_file(outfile)
            text_audio_dur_time = text_audio.duration_seconds * 1000
            
            if i < len(refer_subtitles) - 1:
                nxt_start = refer_subtitles[i+1].start.total_seconds() * 1000
                dur_time =  nxt_start - start_time
            else:
                org_dur_time = audio_seg.duration_seconds * 1000
                dur_time = org_dur_time - start_time
            
            ratio = text_audio_dur_time / dur_time

            if text_audio_dur_time > dur_time:
                if if_aliginment:
                    tmp_audio = self.map_vocal(audio=text_audio,ratio=ratio,dur_time=dur_time,
                                                    wav_name=f"map_{i}_refer.wav",temp_folder=refer_wav_root)
                    tmp_audio += AudioSegment.silent(dur_time - tmp_audio.duration_seconds*1000)
                else:
                    tmp_audio = text_audio
            else:
                tmp_audio = text_audio + AudioSegment.silent(dur_time - text_audio_dur_time)
          
            new_audio_seg += tmp_audio

        infer_audio = os.path.join(out_path, f"{ttime()}_gpt_sovits_refer.wav")
        new_audio_seg.export(infer_audio, format="wav")
        
        return (infer_audio,)
    def map_vocal(self,audio:AudioSegment,ratio:float,dur_time:float,wav_name:str,temp_folder:str):
        tmp_path = f"{temp_folder}/map_{wav_name}"
        audio.export(tmp_path, format="wav")
        
        clone_path = f"{temp_folder}/cloned_{wav_name}"
        reader = audiotsm.io.wav.WavReader(tmp_path)
        
        writer = audiotsm.io.wav.WavWriter(clone_path,channels=reader.channels,
                                        samplerate=reader.samplerate)
        wsloa = audiotsm.wsola(channels=reader.channels,speed=ratio)
        wsloa.run(reader=reader,writer=writer)
        audio_extended = AudioSegment.from_file(clone_path)
        return audio_extended[:dur_time]
    
def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def get_files(end_with="pth",model_type="D"):
    file_list = []
    for filepath,dirnames,filenames in os.walk(os.path.join(parent_directory, "logs")):
        for filename in filenames:
            if filename.endswith(end_with) and model_type in filename:
                tmp_path = os.path.join(filepath, filename)
                name_list = splitall(tmp_path)
                if model_type == "ckpt":
                    file_n = name_list[-4] + '&' + name_list[-1]
                else:
                    file_n = name_list[-3] + '&' + name_list[-1]
                file_list.append(file_n)
    return file_list

class GPT_SOVITS_FT:
    @classmethod
    def INPUT_TYPES(s):
        ft_language_list = ["zh", "en", "ja"]
        return {"required":
                    {"audio": ("AUDIO",),
                     "srt": ("SRT",),
                     "exp_name": ("STRING",{
                         "default": "auto"
                     }),
                     "language":(ft_language_list,{
                         "default": "zh"
                     }),
                     "pretrained_s2G":(get_files('pth','G')+["s2G488k.pth"],{
                         "default": "s2G488k.pth"
                     }),
                     "pretrained_s2D":(get_files('pth','D')+["s2D488k.pth"],{
                         "default": "s2D488k.pth"
                     }),
                     "sovits_batch_size": ("INT",{
                         "min": 1,
                         "max": 40,
                         "step": 1,
                         "default":default_batch_size,
                         "display": "slider" 
                     }),
                     "sovits_total_epoch": ("INT",{
                         "min": 1,
                         "max": 25,
                         "step": 1,
                         "default":8,
                         "display": "slider" 
                     }),
                     "text_low_lr_rate": ("FLOAT",{
                         "min": 0.2,
                         "max": 0.6,
                         "step": 0.05,
                         "default":0.4,
                         "display": "slider" 
                     }),
                     "sovits_save_every_epoch": ("INT",{
                         "min": 1,
                         "max": 25,
                         "step": 1,
                         "default":4,
                         "display": "slider" 
                     }),
                     "if_save_latest_sovits":("BOOLEAN",{
                         "default": True
                     }),
                     "if_save_every_sovits_weights":("BOOLEAN",{
                         "default": True
                     }),
                     "pretrained_s1":(get_files("ckpt","ckpt")+["s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"],{
                         "default": "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
                     }),
                     "gpt_batch_size": ("INT",{
                         "min": 1,
                         "max": 40,
                         "step": 1,
                         "default":default_batch_size,
                         "display": "slider" 
                     }),
                     "gpt_total_epoch": ("INT",{
                         "min": 2,
                         "max": 50,
                         "step": 1,
                         "default":15,
                         "display": "slider" 
                     }),
                     "if_dpo":("BOOLEAN",{
                         "default": False
                     }),
                     "if_save_latest_gpt":("BOOLEAN",{
                         "default": True
                     }),
                     "if_save_every_gpt_weights":("BOOLEAN",{
                         "default": True
                     }),
                     "gpt_save_every_epoch": ("INT",{
                         "min": 1,
                         "max": 50,
                         "step": 1,
                         "default":5,
                         "display": "slider" 
                     }),
                    }
                }
    CATEGORY = "AIFSH_GPT_SOVITS"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = "finetune"

    def finetune(self,audio,srt,exp_name,language,pretrained_s2G,
                 pretrained_s2D,sovits_batch_size,sovits_total_epoch,
                 text_low_lr_rate,sovits_save_every_epoch,if_save_latest_sovits,
                 if_save_every_sovits_weights,pretrained_s1,gpt_batch_size,
                 gpt_total_epoch,if_dpo,if_save_latest_gpt,if_save_every_gpt_weights,
                 gpt_save_every_epoch):
        logs_path = os.path.join(parent_directory,"logs")
        srt_path = folder_paths.get_annotated_filepath(srt)
        audio_path = folder_paths.get_annotated_filepath(audio)
        audio_seg = AudioSegment.from_file(audio_path)
        if pretrained_s2D == "s2D488k.pth":
            pretrained_s2D = os.path.join(weights_path,"s2D488k.pth")
        else:
            pretrained_s2D = pretrained_s2D.split("&")
            pretrained_s2D = os.path.join(logs_path,pretrained_s2D[0],"logs_s2",pretrained_s2D[1])
        if pretrained_s2G == "s2G488k.pth":
            pretrained_s2G = os.path.join(weights_path,"s2G488k.pth")
        else:
            pretrained_s2G = pretrained_s2G.split("&")
            pretrained_s2G = os.path.join(logs_path,pretrained_s2G[0],"logs_s2",pretrained_s2G[1])

        if pretrained_s1 == "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt":
            pretrained_s1 = os.path.join(weights_path,"s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
        else:
            pretrained_s1 = pretrained_s1.split("&")
            pretrained_s1 = os.path.join(logs_path,pretrained_s1[0],"logs_s1/ckpt",pretrained_s1[1])
        
        with open(srt_path, 'r', encoding="utf-8") as file:
            file_content = file.read()
        work_path_list = []
        for i, sub in enumerate(list(SrtPare(file_content))):
            start_time = sub.start.total_seconds() * 1000
            end_time = sub.end.total_seconds() * 1000
            if exp_name == "auto":
                try:
                    text = sub.content[1:]
                    exp_name = f"speaker_{int(sub.content[0])}"
                except:
                    text = sub.content
                    exp_name = "speaker_0"
            else:
                text = sub.content
            work_path = os.path.join(parent_directory,"logs",exp_name)
            if work_path not in work_path_list: work_path_list.append(work_path)
            os.makedirs(work_path, exist_ok=True)

            inp_text = os.path.join(work_path, "annotation.list")
            inp_wav_dir = os.path.join(work_path,"wav")
            os.makedirs(inp_wav_dir, exist_ok=True)
            vocal_path = os.path.join(inp_wav_dir, f"{exp_name}-%04d.wav" % (i+1))
            vocal_seg = audio_seg[start_time:end_time]
            vocal_seg.export(vocal_path, format="wav")
            with open(inp_text, 'a', encoding="utf-8") as w:
                line = f'{vocal_path}|{exp_name}|{language}|{text}\n'
                w.write(line)
        for work_path in work_path_list:
            inp_text = os.path.join(work_path, "annotation.list")
            inp_wav_dir = os.path.join(work_path,"wav")
            exp_name = os.path.basename(work_path)
            open1abc(inp_text,inp_wav_dir,exp_name,pretrained_s2G,work_path)
            open1Ba(batch_size=sovits_batch_size,total_epoch=sovits_total_epoch,
                    exp_name=exp_name,text_low_lr_rate=text_low_lr_rate,
                    if_save_latest=if_save_latest_sovits,if_save_every_weights=if_save_every_sovits_weights,
                    save_every_epoch=sovits_save_every_epoch,pretrained_s2G=pretrained_s2G,
                    pretrained_s2D=pretrained_s2D,work_path=work_path)
            open1Bb(batch_size=gpt_batch_size,total_epoch=gpt_total_epoch,exp_name=exp_name,
                    if_dpo=if_dpo,if_save_latest=if_save_latest_gpt,if_save_every_weights=if_save_every_gpt_weights,
                    save_every_epoch=gpt_save_every_epoch,pretrained_s1=pretrained_s1,work_path=work_path)
        return {"ui":{"finetune":[SoVITS_weight_root,GPT_weight_root]}}


class PreViewAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"audio": ("AUDIO",),}
                }

    CATEGORY = "AIFSH_GPT_SOVITS"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_name = os.path.basename(audio)
        tmp_path = os.path.dirname(audio)
        audio_root = os.path.basename(tmp_path)
        return {"ui": {"audio":[audio_name,audio_root]}}

class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["wav", "mp3","WAV","flac","m4a"]]
        return {"required":
                    {"audio": (sorted(files),)},
                }

    CATEGORY = "AIFSH_GPT_SOVITS"

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        return (audio_path,)

class LoadSRT:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["srt", "txt"]]
        return {"required":
                    {"srt": (sorted(files),)},
                }

    CATEGORY = "AIFSH_GPT_SOVITS"

    RETURN_TYPES = ("SRT",)
    FUNCTION = "load_srt"

    def load_srt(self, srt):
        srt_path = folder_paths.get_annotated_filepath(srt)
        return (srt_path,)