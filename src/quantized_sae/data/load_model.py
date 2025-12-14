from huggingface_hub import hf_hub_download

# Repo+path you gave
# repo_id = "ctigges/pythia-70m-deduped__res-sm_processed"
repo_id = "EleutherAI/sae-pythia-70m-32k"
# subfolder = "3-res-sm"
subfolder = "layers.3"
filename = "sae.safetensors"

# Download to HF cache; returns the local path to the file
local_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    subfolder=subfolder,   # path inside the repo
    revision="main",       # or a commit hash if you want to pin it
    # token="hf_..."       # uncomment if the repo requires auth
)

print("Saved to cache at:", local_path)

# (Optional) copy it wherever you want outside the cache:
import os, shutil
dest = "./baseline_SAE/"+repo_id+"/"+subfolder+"/"+filename
os.makedirs(os.path.dirname(dest), exist_ok=True)
shutil.copy2(local_path, dest)
print("Copied to:", dest)