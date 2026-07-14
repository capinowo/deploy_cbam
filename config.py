"""
Konfigurasi model untuk aplikasi perbandingan YOLOv12n + CBAM ablation study.
Semua model diambil dari HuggingFace Hub milik capinowo.
"""

HF_USERNAME = "capinowo"

# label tampilan -> slug repo HuggingFace
VARIANT_SLUGS = {
    "Baseline": "baseline",
    "CBAM - Channel Only": "cbam-channel",
    "CBAM - Spatial Only": "cbam-spatial",
    "CBAM - Full (Channel + Spatial)": "cbam-full",
}

EPOCHS = [20, 35, 50]


def _build_repo_id(slug: str, epoch: int) -> str:
    return f"{HF_USERNAME}/skripsi-yolo12n-{slug}-checkpoint-{epoch}e"


# dict lengkap, contoh: "Baseline (20 epoch)" -> "capinowo/skripsi-yolo12n-baseline-checkpoint-20e"
MODEL_REGISTRY = {}
for _label, _slug in VARIANT_SLUGS.items():
    for _epoch in EPOCHS:
        _key = f"{_label} ({_epoch} epoch)"
        MODEL_REGISTRY[_key] = _build_repo_id(_slug, _epoch)

# warna bounding box per varian (BGR), biar gampang dibedain kalau beberapa model ditampilkan bareng
VARIANT_COLORS = {
    "baseline": (60, 60, 220),        # merah
    "cbam-channel": (60, 200, 60),    # hijau
    "cbam-spatial": (220, 160, 40),   # oranye
    "cbam-full": (220, 60, 200),      # pink/ungu
}


def get_color_for_label(label: str):
    repo_id = MODEL_REGISTRY.get(label, "")
    for slug, color in VARIANT_COLORS.items():
        if slug in repo_id:
            return color
    return (255, 255, 255)