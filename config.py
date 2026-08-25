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

# repo checkpoint hasil finetuning (lanjutan dari checkpoint 50 epoch, +75 epoch lagi)
# key di sini pakai underscore biar gampang dicocokin ke VARIANT_SLUGS (yang pakai dash)
FINETUNED_REPOS = {
    "baseline": "capinowo/skripsi-yolo12n-baseline-checkpoint-50e-finetuned-75e",
    "cbam_channel": "capinowo/skripsi-yolo12n-cbam-channel-checkpoint-50e-finetuned-75e",
    "cbam_spatial": "capinowo/skripsi-yolo12n-cbam-spatial-checkpoint-50e-finetuned-75e",
    "cbam_full": "capinowo/skripsi-yolo12n-cbam-full-checkpoint-50e-finetuned-75e",
}

# gabungin ke MODEL_REGISTRY, contoh:
# "Baseline (Finetuned 75 epoch)" -> "capinowo/skripsi-yolo12n-baseline-checkpoint-50e-finetuned-75e"
_SLUG_TO_LABEL = {slug.replace("-", "_"): label for label, slug in VARIANT_SLUGS.items()}
for _ft_key, _ft_repo in FINETUNED_REPOS.items():
    _ft_label = _SLUG_TO_LABEL.get(_ft_key, _ft_key)
    MODEL_REGISTRY[f"{_ft_label} (Finetuned 75 epoch)"] = _ft_repo

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