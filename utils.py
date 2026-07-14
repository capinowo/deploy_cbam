import cv2
import numpy as np
from PIL import Image

from config import get_color_for_label


def convert_uploaded_image(uploaded_file) -> np.ndarray:
    """Ubah file upload Streamlit jadi array BGR (format OpenCV)."""
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


def draw_detections(frame_bgr: np.ndarray, result, label: str) -> np.ndarray:
    """Gambar bounding box dari 1 hasil prediksi ultralytics ke frame."""
    color = get_color_for_label(label)
    if result.boxes is None or len(result.boxes) == 0:
        return frame_bgr

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    track_ids = None
    if getattr(result.boxes, "id", None) is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)

    for i, (box, conf) in enumerate(zip(boxes, confs)):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        tag = f"ID{track_ids[i]} " if track_ids is not None else ""
        text = f"{tag}{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame_bgr, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame_bgr


def summarize_result(result) -> dict:
    """Hitung jumlah deteksi, confidence rata-rata & maksimum dari 1 hasil prediksi."""
    if result.boxes is None or len(result.boxes) == 0:
        return {"n_det": 0, "avg_conf": 0.0, "max_conf": 0.0}
    confs = result.boxes.conf.cpu().numpy().tolist()
    return {
        "n_det": len(confs),
        "avg_conf": float(sum(confs) / len(confs)),
        "max_conf": float(max(confs)),
    }


def open_video_writer(output_path: str, fps: float, frame_size: tuple):
    """Buat VideoWriter pakai codec mp4v (gak bergantung H.264/OpenH264 yang sering gak ada di server)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if writer.isOpened():
        return writer
    return None