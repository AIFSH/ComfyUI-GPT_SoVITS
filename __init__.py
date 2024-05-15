import site
import os,sys
import logging
from server import PromptServer

now_dir = os.path.dirname(os.path.abspath(__file__))
site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if(site_packages_roots==[]):site_packages_roots=["%s/runtime/Lib/site-packages" % now_dir]
#os.environ["OPENBLAS_NUM_THREADS"] = "4"
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/GPT_SoVITS.pth" % (site_packages_root), "w") as f:
                f.write(
                    "%s\n%s/GPT_SoVITS\n%s/GPT_SoVITS/text\n"
                    % (now_dir,now_dir,now_dir)
                )
            break
        except PermissionError:
            raise PermissionError

if os.path.isfile("%s/GPT_SoVITS.pth" % (site_packages_root)):
    print("!!!GPT_SoVITS path was added to " + "%s/GPT_SoVITS.pth" % (site_packages_root) 
    + "\n if meet `No module` error,try `python main.py` again, don't be foolish to pip install tools")

from huggingface_hub import snapshot_download
model_path = os.path.join(now_dir,"pretrained_models")
if not os.path.isfile(os.path.join(model_path,"s2G488k.pth")):
    snapshot_download(repo_id="lj1995/GPT-SoVITS",local_dir=model_path)
else:
    print("GPT_SoVITS use cache models,make sure your 'pretrained_models' complete")

WEB_DIRECTORY = "./web"
from .nodes import LoadSRT,LoadAudio, GPT_SOVITS_INFER, PreViewAudio,GPT_SOVITS_FT, GPT_SOVITS_TTS

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "GPT_SOVITS_FT": GPT_SOVITS_FT,
    "LoadAudio": LoadAudio,
    "PreViewAudio": PreViewAudio,
    "LoadSRT": LoadSRT,
    "GPT_SOVITS_INFER": GPT_SOVITS_INFER,
    "GPT_SOVITS_TTS": GPT_SOVITS_TTS
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT_SOVITS_FT": "GPT_SOVITS Finetune",
    "LoadAudio": "AudioLoader",
    "PreViewAudio": "PreView Audio",
    "LoadSRT": "SRT FILE Loader",
    "GPT_SOVITS_INFER": "GPT_SOVITS Inference",
    "GPT_SOVITS_TTS": "GPT_SOVITS TTS"
}

@PromptServer.instance.routes.get("/gpt_sovits/reboot")
def restart(self):
    try:
        sys.stdout.close_log()
    except Exception as e:
        pass

    return os.execv(sys.executable, [sys.executable] + sys.argv)
