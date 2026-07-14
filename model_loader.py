"""
Download & load model YOLOv12n (+CBAM) dari HuggingFace Hub, dengan caching
supaya Streamlit gak download & load ulang tiap kali user interaksi/rerun.
"""

import streamlit as st
from huggingface_hub import hf_hub_download, list_repo_files
from ultralytics import YOLO

from cbam_modules import register_custom_classes_in_main


def _find_weight_filename(repo_id: str) -> str:
    files = list_repo_files(repo_id)
    pt_files = [f for f in files if f.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(
            f"Gak nemu file .pt di repo '{repo_id}'. Cek lagi isi repo HuggingFace-nya."
        )
    for preferred in ("best.pt", "last.pt"):
        for f in pt_files:
            if f.endswith(preferred):
                return f
    return pt_files[0]


@st.cache_resource(show_spinner=False)
def load_model(repo_id: str) -> YOLO:
    register_custom_classes_in_main()
    weight_filename = _find_weight_filename(repo_id)
    local_path = hf_hub_download(repo_id=repo_id, filename=weight_filename)
    model = YOLO(local_path)
    return model