import os
import shutil
import subprocess
import modal

# ---------- Base image + deps ----------
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
    .run_commands("comfy --skip-prompt install --fast-deps --nvidia --version 0.3.47")
)

# ---------- Custom nodes ----------
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

# ---------- Build-time HF models (Flux1-Kontext + CLIP encoders) & cleanup ----------
def prepare_models():
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

    # --- Keep: Flux1-Kontext UNet (HF) ---
    flux_unet_cached = hf_hub_download(
        repo_id="6chan/flux1-kontext-dev-fp8",
        filename="flux1-kontext-dev-fp8-e4m3fn.safetensors",
        cache_dir="/cache",
    )
    flux_unet_dest = os.path.join(dirs["unet"], "flux1-kontext-dev-fp8-e4m3fn.safetensors")
    subprocess.run(["ln", "-sf", flux_unet_cached, flux_unet_dest], check=True)
    print(f"[prepare_models] Installed: {flux_unet_dest}")

    # --- NEW: Download CLIP text encoders from HF (NOT Drive) ---
    clip_repo = "comfyanonymous/flux_text_encoders"

    clip_l_cached = hf_hub_download(
        repo_id=clip_repo,
        filename="clip_l.safetensors",
        cache_dir="/cache",
    )
    clip_l_dest = os.path.join(dirs["clip"], "clip_l.safetensors")
    subprocess.run(["ln", "-sf", clip_l_cached, clip_l_dest], check=True)
    print(f"[prepare_models] Installed: {clip_l_dest}")

    t5xxl_cached = hf_hub_download(
        repo_id=clip_repo,
        filename="t5xxl_fp16.safetensors",
        cache_dir="/cache",
    )
    t5xxl_dest = os.path.join(dirs["clip"], "t5xxl_fp16.safetensors")
    subprocess.run(["ln", "-sf", t5xxl_cached, t5xxl_dest], check=True)
    print(f"[prepare_models] Installed: {t5xxl_dest}")

    # --- Remove unwanted WAN/T5/VAE/LightX2V and stale aliases (we keep ae/clip_l/t5xxl in right folders) ---
    to_remove = [
        os.path.join(dirs["diff"], "wan2.2_i2v_high_noise_14B_fp16.safetensors"),
        os.path.join(dirs["diff"], "wan2.2_i2v_low_noise_14B_fp16.safetensors"),
        os.path.join(dirs["vae"], "wan_2.1_vae.safetensors"),
        os.path.join(dirs["te"], "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
        os.path.join(dirs["loras"], "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"),
        os.path.join(dirs["loras"], "ightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"),  # typo guard
        # Kill any older misplaced aliases:
        os.path.join(dirs["te"], "clip_l.safetensors"),
        os.path.join(dirs["te"], "t5xxl_fp16.safetensors"),
    ]
    for path in to_remove:
        try:
            if os.path.lexists(path):
                os.remove(path)
                print(f"[prepare_models] Removed: {path}")
        except Exception as e:
            print(f"[prepare_models] Skip remove {path}: {e}")

# ---------- Runtime: pull Drive assets with fallback & route ----------
def download_gdrive_assets():
    import gdown

    base = "/root/comfy/ComfyUI/models"
    clip_dir = os.path.join(base, "clip")
    vae_dir = os.path.join(base, "vae")
    lora_dir = os.path.join(base, "loras")
    for d in (clip_dir, vae_dir, lora_dir):
        os.makedirs(d, exist_ok=True)

    staging = "/tmp/gdrive_downloads"
    os.makedirs(staging, exist_ok=True)

    file_ids = [
        "1-3zuAumzikFF__3pvBUD_2nNb_wpyYBy",   # JD3sNDFFK.safetensors (LoRA)
        "1xrb9IqFx8vv1qGs_Ac0gIhFk7t_3-AyD",
        "1iTP9J37AXoJMl87Z7LttgE67wI9VV5oe",
        "10zGbW5o-ExwP8HVM-gQC8avH9ZICIOS-",   # trailing '-': keep as provided
    ]

    def route_dir(filename: str) -> str:
        name = filename.lower()
        if name == "ae.safetensors":
            return vae_dir
        # We DO NOT install clip_l/t5xxl from Drive anymore; those come from HF.
        return lora_dir

    def try_download(fid: str):
        """Try multiple methods; return absolute output path or None."""
        cwd = os.getcwd()
        os.chdir(staging)
        out_path = None
        try:
            # 1) ID-based
            out_path = gdown.download(id=fid, quiet=False)
            if out_path:
                return os.path.abspath(out_path)
            # 2) URL fallbacks
            url_variants = [
                f"https://drive.google.com/uc?id={fid}&confirm=t",
                f"https://drive.google.com/open?id={fid}",
                f"https://drive.google.com/file/d/{fid}/view?usp=sharing",
            ]
            for url in url_variants:
                out_path = gdown.download(url=url, quiet=False, fuzzy=True)
                if out_path:
                    return os.path.abspath(out_path)
            return None
        finally:
            os.chdir(cwd)

    for fid in file_ids:
        fid = fid.strip()
        try:
            print(f"[gdrive] Downloading file_id={fid}")
            out_path = try_download(fid)
            if not out_path:
                print(f"[gdrive] Failed: file_id={fid} (all methods)")
                continue

            fname = os.path.basename(out_path)
            src = out_path
            if not os.path.exists(src) or os.path.getsize(src) == 0:
                print(f"[gdrive] Empty/missing: {src}")
                continue

            # If Drive returns clip_l/t5xxl, ignore & delete (we use HF versions).
            lower = fname.lower()
            if lower in ("clip_l.safetensors", "t5xxl_fp16.safetensors"):
                try:
                    os.remove(src)
                except Exception:
                    pass
                print(f"[gdrive] Skipped installing {fname} from Drive (using HF copy).")
                continue

            dest_dir = route_dir(fname)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, fname)

            if os.path.exists(dest) or os.path.islink(dest):
                try:
                    os.remove(dest)
                except Exception:
                    pass

            shutil.move(src, dest)
            print(f"[gdrive] Saved -> {dest} ({os.path.getsize(dest)} bytes)")
        except Exception as e:
            print(f"[gdrive] Error downloading {fid}: {e}")

# ---------- Build cache for HF ----------
vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
image = (
    image.pip_install("huggingface_hub[hf_transfer]>=0.34.0,<1.0")
    .run_function(prepare_models, volumes={"/cache": vol})
)

# ---------- App ----------
app = modal.App(name="example-comfyui", image=image)

@app.function(
    max_containers=1,
    gpu="L40S",
    volumes={"/cache": vol},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=60)
def ui():
    download_gdrive_assets()
    subprocess.Popen(["bash", "-lc", "comfy launch -- --listen 0.0.0.0 --port 8000"], shell=False)
