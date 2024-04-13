import os
import torch
import psutil
import traceback
import folder_paths
import platform,signal
from config import python_exec
from tools import my_utils
from subprocess import Popen
from .inference import is_half,bert_path,parent_directory,cnhubert_base_path
out_path = folder_paths.get_output_directory()
if_gpu_ok = False
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
# 判断是否有能用来训练和加速推理的N卡
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper()for value in ["10","16","20","30","40","A2","A3","A4","P4","A50","500","A60","70","80","90","M4","T4","TITAN","L4","4060"]):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(int(torch.cuda.get_device_properties(i).total_memory/ 1024/ 1024/ 1024+ 0.4))

if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = ("%s\t%s" % ("0", "CPU"))
    gpu_infos.append("%s\t%s" % ("0", "CPU"))
    default_batch_size = psutil.virtual_memory().total/ 1024 / 1024 / 1024 / 2
gpus = "-".join([i[0] for i in gpu_infos])

ps1abc=[]
root_py_path = os.path.join(parent_directory,"GPT_SoVITS")
pretrained_s2G_path = os.path.join(parent_directory,"pretrained_models","s2G488k.pth")

gpu_numbers1a = "%s-%s"%(gpus,gpus)
gpu_numbers1Ba = "%s-%s"%(gpus,gpus)
gpu_numbers1c = "%s-%s"%(gpus,gpus)
def open1abc(inp_text,inp_wav_dir,exp_name,pretrained_s2G,work_path):
    global ps1abc
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    dataset_py_path = os.path.join(root_py_path, "prepare_datasets")
    if (ps1abc == []):
        opt_dir = work_path
        try:
            #############################1a
            path_text="%s/2-name2text.txt" % opt_dir
            if(os.path.exists(path_text)==False or (os.path.exists(path_text)==True and len(open(path_text,"r",encoding="utf8").read().strip("\n").split("\n"))<2)):
                config={
                    "inp_text":inp_text,
                    "inp_wav_dir":inp_wav_dir,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "bert_pretrained_dir":bert_path,
                    "is_half": str(is_half)
                }
                gpu_names=gpu_numbers1a.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    cmd = f'{python_exec} {dataset_py_path}/1-get-text.py'
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                print("进度：1a-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True})
                for p in ps1abc:p.wait()

                opt = []
                for i_part in range(all_parts):#txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")

            print("进度：1a-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True})
            ps1abc=[]
            #############################1b
            config={
                "inp_text":inp_text,
                "inp_wav_dir":inp_wav_dir,
                "exp_name":exp_name,
                "opt_dir":opt_dir,
                "cnhubert_base_dir":cnhubert_base_path,
            }
            gpu_names=gpu_numbers1Ba.split("-")
            all_parts=len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                    }
                )
                os.environ.update(config)
                cmd = f'{python_exec} {dataset_py_path}/2-get-hubert-wav32k.py'
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            print("进度：1a-done, 1b-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}) 
            for p in ps1abc:p.wait()
            print("进度：1a1b-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}) 
            ps1abc=[]
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if(os.path.exists(path_semantic)==False or (os.path.exists(path_semantic)==True and os.path.getsize(path_semantic)<31)):
                config={
                    "inp_text":inp_text,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "pretrained_s2G":pretrained_s2G,
                    "s2config_path":f"{root_py_path}/configs/s2.json",
                }
                gpu_names=gpu_numbers1c.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    cmd = f'{python_exec} {dataset_py_path}/3-get-semantic.py'
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                print("进度：1a1b-done, 1cing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}) 
                for p in ps1abc:p.wait()

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                print("进度：all-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}) 
            ps1abc = []
            print("一键三连进程结束", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}) 
        except:
            traceback.print_exc()
            close1abc()
            print("一键三连中途报错", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}) 
    else:
        print("已有正在进行的一键三连任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}) 

def kill_proc_tree(pid, including_parent=True):  
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
system=platform.system()
def kill_process(pid):
    if(system=="Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)

def close1abc():
    global ps1abc
    if (ps1abc != []):
        for p1abc in ps1abc:
            try:
                kill_process(p1abc.pid)
            except:
                traceback.print_exc()
        ps1abc=[]
    return "已终止所有一键三连进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

gpu_numbers1Ba = "%s" % (gpus)
gpu_numbers1Bb = "%s" % (gpus)

import json
import yaml
p_train_SoVITS=None
def open1Ba(batch_size,total_epoch,exp_name,text_low_lr_rate,if_save_latest,if_save_every_weights,save_every_epoch,pretrained_s2G,pretrained_s2D,work_path):
    SoVITS_weight_root = os.path.join(out_path,"sovits_weights")
    os.makedirs(SoVITS_weight_root, exist_ok=True)
    global p_train_SoVITS
    if(p_train_SoVITS==None):
        with open(f"{root_py_path}/configs/s2.json")as f:
            data=f.read()
            data=json.loads(data)
        s2_dir = work_path
        os.makedirs("%s/logs_s2"%(s2_dir),exist_ok=True)
        if(is_half==False):
            data["train"]["fp16_run"]=False
            batch_size=max(1,batch_size//2)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["train"]["text_low_lr_rate"]=text_low_lr_rate
        data["train"]["pretrained_s2G"]=pretrained_s2G
        data["train"]["pretrained_s2D"]=pretrained_s2D
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["save_every_epoch"]=save_every_epoch
        data["train"]["gpu_numbers"]=gpu_numbers1Ba
        data["data"]["exp_dir"]=data["s2_ckpt_dir"]=s2_dir
        data["save_weight_dir"]=SoVITS_weight_root
        data["name"]=exp_name
        tmp_config_path="%s/tmp_s2.json"%s2_dir
        with open(tmp_config_path,"w")as f:f.write(json.dumps(data))

        cmd = f'{python_exec} {root_py_path}/s2_train.py --config {tmp_config_path}'
        print("SoVITS训练开始：%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}) 
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS=None
        print("SoVITS训练完成",{"__type__":"update","visible":True},{"__type__":"update","visible":False})
    else:
        print("已有正在进行的SoVITS训练任务，需先终止才能开启下一次任务",{"__type__":"update","visible":False},{"__type__":"update","visible":True})

def close1Ba():
    global p_train_SoVITS
    if(p_train_SoVITS!=None):
        kill_process(p_train_SoVITS.pid)
        p_train_SoVITS=None
    return "已终止SoVITS训练",{"__type__":"update","visible":True},{"__type__":"update","visible":False}

p_train_GPT=None
gpu_numbers = "%s" % (gpus)
def open1Bb(batch_size,total_epoch,exp_name,if_dpo,if_save_latest,if_save_every_weights,save_every_epoch,pretrained_s1,work_path):
    GPT_weight_root = os.path.join(out_path,"gpt_weights")
    global p_train_GPT
    if(p_train_GPT==None):
        with open(f"{root_py_path}/configs/s1longer.yaml")as f:
            data=f.read()
            data=yaml.load(data, Loader=yaml.FullLoader)
        s1_dir=work_path
        os.makedirs("%s/logs_s1"%(s1_dir),exist_ok=True)
        if(is_half==False):
            data["train"]["precision"]="32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["pretrained_s1"]=pretrained_s1
        data["train"]["save_every_n_epoch"]=save_every_epoch
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["if_dpo"]=if_dpo
        data["train"]["half_weights_save_dir"]=GPT_weight_root
        data["train"]["exp_name"]=exp_name
        data["train_semantic_path"]="%s/6-name2semantic.tsv"%s1_dir
        data["train_phoneme_path"]="%s/2-name2text.txt"%s1_dir
        data["output_dir"]="%s/logs_s1"%s1_dir

        os.environ["_CUDA_VISIBLE_DEVICES"]=gpu_numbers.replace("-",",")
        os.environ["hz"]="25hz"
        tmp_config_path="%s/tmp_s1.yaml"%s1_dir
        with open(tmp_config_path, "w") as f:f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = f'{python_exec} {root_py_path}/s1_train.py --config_file {tmp_config_path}'
        print("GPT训练开始：%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}) 
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT=None
        print("GPT训练完成",{"__type__":"update","visible":True},{"__type__":"update","visible":False}) 
    else:
        print("已有正在进行的GPT训练任务，需先终止才能开启下一次任务",{"__type__":"update","visible":False},{"__type__":"update","visible":True}) 

def close1Bb():
    global p_train_GPT
    if(p_train_GPT!=None):
        kill_process(p_train_GPT.pid)
        p_train_GPT=None
    return "已终止GPT训练",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
