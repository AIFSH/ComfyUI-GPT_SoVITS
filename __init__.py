import site
import os,sys

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
            with open("%s/users.pth" % (site_packages_root), "w") as f:
                f.write(
                    "%s\n%s/GPT_SoVITS\n"
                    % (now_dir,now_dir)
                )
            break
        except PermissionError:
            pass

from huggingface_hub import snapshot_download
snapshot_download(repo_id="lj1995/GPT-SoVITS",local_dir=os.path.join(now_dir,"pretrained_models"))

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
