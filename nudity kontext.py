import subprocess
import os
import shutil
import modal

# Base image + deps
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "build-essential",
        "cmake",
        "wget",
    )
    .pip_install(
        "gguf",
        "llama-cpp-python",
        "opencv-python-headless",
        "imageio[ffmpeg]",
        "moviepy",
        "fastapi[standard]==0.115.4",
        "comfy-cli==1.5.1",
        "gdown",  # Google Drive downloads
    )
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.47"
    )
)

# Custom nodes
image = image.run_commands(
    "comfy node install --fast-deps was-node-suite-comfyui@1.0.2",
    "git clone https://github.com/ChenDarYen/ComfyUI-NAG.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-NAG",
    "git clone https://github.com/kijai/ComfyUI-KJNodes.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    "git clone https://github.com/cubiq/ComfyUI_essentials.git /root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials",
    "git clone https://github.com/city96/ComfyUI-GGUF.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-GGUF",
    "git clone https://github.com/rgthree/rgthree-comfy.git /root/comfy/ComfyUI/custom_nodes/rgthree-comfy",
    "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    "git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation",
)

def prepare_models():
    """
    - Keep: Flux1-Kontext UNet (Hugging Face)
    - Remove: WAN/T5/VAE/LightX2V weights and previous aliases
    """
    from huggingface_hub import hf_hub_download

    base = "/root/comfy/ComfyUI/models"
    dirs = {
        "unet": os.path.join(base, "unet"),
        "diff": os.path.join(base, "diffusion_models"),
        "vae": os.path.join(base, "vae"),
        "te": os.path.join(base, "text_encoders"),
        "clip": os.path.join(base, "clip"),
        "loras": os.path.join(base, "loras"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # --- Keep: Flux1-Kontext UNet ---
    flux_unet = hf_hub_download(
        repo_id="6chan/flux1-kontext-dev-fp8",
        filename="flux1-kontext-dev-fp8-e4m3fn.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {flux_unet} {os.path.join(dirs['unet'], 'flux1-kontext-dev-fp8-e4m3fn.safetensors')}",
        shell=True,
        check=True,
    )
    print("[prepare_models] Flux1-Kontext UNet installed.")

    # --- Remove unwanted models if present ---
    to_remove = [
        os.path.join(dirs["diff"], "wan2.2_i2v_high_noise_14B_fp16.safetensors"),
        os.path.join(dirs["diff"], "wan2.2_i2v_low_noise_14B_fp16.safetensors"),
        os.path.join(dirs["vae"],  "wan_2.1_vae.safetensors"),
        os.path.join(dirs["te"],   "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
        os.path.join(dirs["loras"],"lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"),
        os.path.join(dirs["loras"],"ightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"),  # typo guard
        # Remove any old aliases created earlier (if any)
        os.path.join(dirs["te"], "clip_l.safetensors"),
        os.path.join(dirs["te"], "t5xxl_fp16.safetensors"),
        os.path.join(dirs["vae"], "ae.safetensors"),
    ]
    for path in to_remove:
        try:
            if os.path.islink(path) or os.path.isfile(path):
                os.remove(path)
                print(f"[prepare_models] Removed: {path}")
        except Exception as e:
            print(f"[prepare_models] Skipped removing {path}: {e}")

def download_gdrive_assets():
    """
    Download multiple Google Drive files by ID, preserve original filenames,
    and route to the correct ComfyUI model subfolder:

    - clip_l.safetensors        -> models/clip
    - t5xxl_fp16.safetensors    -> models/clip
    - ae.safetensors            -> models/vae
    - anything else (.safetensors) -> models/loras
    """
    import gdown

    base = "/root/comfy/ComfyUI/models"
    clip_dir = os.path.join(base, "clip")
    vae_dir  = os.path.join(base, "vae")
    lora_dir = os.path.join(base, "loras")
    for d in (clip_dir, vae_dir, lora_dir):
        os.makedirs(d, exist_ok=True)

    staging = "/tmp/gdrive_downloads"
    os.makedirs(staging, exist_ok=True)

    file_ids = [
        "1-3zuAumzikFF__3pvBUD_2nNb_wpyYBy",          # JD3sNDFFK.safetensors (LoRA)
        "1xrb9IqFx8vv1qGs_Ac0gIhFk7t_3-AyD",          # new
        "1iTP9J37AXoJMl87Z7LttgE67wI9VV5oe",          # new
        "10zGbW5o-ExwP8HVM-gQC8avH9ZICIOS-",          # new
    ]

    def _route(filename: str) -> str:
        name = filename.lower()
        if name == "clip_l.safetensors":
            return clip_dir
        if name == "t5xxl_fp16.safetensors":
            return clip_dir
        if name == "ae.safetensors":
            return vae_dir
        return lora_dir

    for fid in file_ids:
        try:
            print(f"[gdrive] Downloading file_id={fid}")
            cwd = os.getcwd()
            os.chdir(staging)
            out = gdown.download(id=fid, quiet=False)  # saves with original filename
            os.chdir(cwd)

            if not out:
                print(f"[gdrive] Failed to download file_id={fid} (no output path).")
                continue

            fname = os.path.basename(out)
            src = os.path.join(staging, fname)
            if not os.path.exists(src) or os.path.getsize(src) == 0:
                print(f"[gdrive] Downloaded file missing/empty: {src}")
                continue

            dest_dir = _route(fname)
            dest = os.path.join(dest_dir, fname)
            if os.path.exists(dest):
                try:
                    os.remove(dest)
                except Exception:
                    pass
            shutil.move(src, dest)
            print(f"[gdrive] Saved -> {dest} ({os.path.getsize(dest)} bytes)")

        except Exception as e:
            print(f"[gdrive] Error downloading {fid}: {e}")

# Cache HF artifacts between builds
vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    image.pip_install("huggingface_hub[hf_transfer]>=0.34.0,<1.0")
    .run_function(prepare_models, volumes={"/cache": vol})
)

app = modal.App(name="example-comfyui", image=image)

@app.function(
    max_containers=1,
    gpu="L40S",
    volumes={"/cache": vol},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=60)
def ui():
    # Pull all Google Drive assets and route them
    download_gdrive_assets()

    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
