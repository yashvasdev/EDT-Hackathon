"""
Modal deployment for BADAS collision prediction via WebSocket.

Usage:
    modal serve app.py      # Dev with live reload
    modal deploy app.py     # Production deployment
"""

from pathlib import Path

import modal
from modal import App, Image, asgi_app, concurrent, enter

app = App("badas-collision-prediction")

project_dir = Path(__file__).parent

huggingface_secret = modal.Secret.from_name("huggingface-secret")


def download_models():
    from huggingface_hub import hf_hub_download, snapshot_download

    hf_hub_download(repo_id="nexar-ai/badas-open", filename="weights/badas_open.pth")
    snapshot_download(repo_id="facebook/vjepa2-vitl-fpc16-256-ssv2")


image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "torch==2.5.0",
        "torchvision==0.20.0",
        "transformers>=4.40.0",
        "huggingface_hub>=0.20.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "albumentations>=1.3.0",
        "psutil>=5.9.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "fastapi[standard]",
    )
    .run_function(download_models, secrets=[huggingface_secret])
    .add_local_dir(
        project_dir / ".." / "badas-uv" / "badas",
        remote_path="/root/badas",
    )
    .add_local_file(
        project_dir / "server.py",
        remote_path="/root/server.py",
    )
)


@app.cls(
    image=image,
    gpu="L4",
    scaledown_window=300,
    secrets=[huggingface_secret],
)
@concurrent(max_inputs=10)
class BADASService:
    @enter()
    def load_model(self):
        import sys
        from huggingface_hub import whoami
        from badas import load_badas_model

        sys.path.insert(0, "/root")

        print("I AM", whoami())

        self.model = load_badas_model(device="cuda")
        self.device = "cuda"

    @asgi_app()
    def serve(self):
        import sys

        sys.path.insert(0, "/root")

        from server import create_app

        return create_app(self.model, self.device)
