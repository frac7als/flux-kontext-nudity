import subprocess
import os
import modal

# It's good practice to list dependencies in a structured way
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "build-essential",
        "cmake",
        "wget",
    )
    # Consolidated list of Python dependencies
    .pip_install(
        "gguf",
        "llama-cpp-python",
        "opencv-python-headless",
        "imageio[ffmpeg]",
        "moviepy",
        "fastapi[standard]==0.115.4",
        "comfy-cli==1.5.1",
        "gdown",  # for Google Drive downloads
    )
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.47"
    )
)

# ## Downloading custom nodes
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

def hf_download():
    from huggingface_hub import hf_hub_download

    # ---------------- WAN 2.2 models ----------------
    wan_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {wan_model} /root/comfy/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
        shell=True,
        check=True,
    )

    wan_model_low_noise = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {wan_model_low_noise} /root/comfy/ComfyUI/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
        shell=True,
        check=True,
    )

    vae_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/vae/wan_2.1_vae.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {vae_model} /root/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors",
        shell=True,
        check=True,
    )

    t5_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {t5_model} /root/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        shell=True,
        check=True,
    )

    # ======================================================================
    # CONFIRMED FIX FOR THE MISSING LORA (LightX2V)
    # ======================================================================
    lora_dir = "/root/comfy/ComfyUI/models/loras"
    os.makedirs(lora_dir, exist_ok=True)

    lora_model_path = hf_hub_download(
        repo_id="Kijai/WanVideo_comfy",
        filename="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        cache_dir="/cache",
    )
    lora_filename = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
    subprocess.run(
        f"ln -sf {lora_model_path} {os.path.join(lora_dir, lora_filename)}",
        shell=True,
        check=True,
    )

    # ======================================================================
    # EXTRA: Flux1-Kontext UNet into ComfyUI/models/unet
    # ======================================================================
    unet_dir = "/root/comfy/ComfyUI/models/unet"
    os.makedirs(unet_dir, exist_ok=True)

    flux_unet = hf_hub_download(
        repo_id="6chan/flux1-kontext-dev-fp8",
        filename="flux1-kontext-dev-fp8-e4m3fn.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {flux_unet} {os.path.join(unet_dir, 'flux1-kontext-dev-fp8-e4m3fn.safetensors')}",
        shell=True,
        check=True,
    )

def maybe_download_gdrive_lora():
    """
    Download JD3sNDFFK.safetensors from Google Drive into ComfyUI/models/loras
    """
    import gdown

    file_id = "1-3zuAumzikFF__3pvBUD_2nNb_wpyYBy"  # <-- your exact file ID
    lora_dir = "/root/comfy/ComfyUI/models/loras"
    os.makedirs(lora_dir, exist_ok=True)
    dest_path = os.path.join(lora_dir, "JD3sNDFFK.safetensors")

    try:
        print(f"[gdrive] Downloading JD3sNDFFK.safetensors from file_id={file_id}")
        gdown.download(id=file_id, output=dest_path, quiet=False)

        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            raise RuntimeError("Downloaded file missing or empty")
        print(f"[gdrive] LoRA saved to {dest_path}")
    except Exception as e:
        print(f"[gdrive] Failed to download LoRA: {e}")

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    # install huggingface_hub with hf_transfer support to speed up downloads
    image.pip_install("huggingface_hub[hf_transfer]>=0.34.0,<1.0")
    .run_function(
        hf_download,
        # persist the HF cache to a Modal Volume so future runs don't re-download models
        volumes={"/cache": vol},
    )
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
    # Download your Google Drive LoRA
    maybe_download_gdrive_lora()

    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
