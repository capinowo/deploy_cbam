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


def parse_yolo_label(label_bytes: bytes, img_width: int, img_height: int) -> list:
    """Parse file label YOLO txt (class cx cy w h, normalized 0-1) jadi list box xyxy piksel."""
    boxes = []
    text = label_bytes.decode("utf-8", errors="ignore")
    for line in text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        x1 = (cx - w / 2) * img_width
        y1 = (cy - h / 2) * img_height
        x2 = (cx + w / 2) * img_width
        y2 = (cy + h / 2) * img_height
        boxes.append({"class_id": cls_id, "box": [x1, y1, x2, y2]})
    return boxes


def draw_gt_boxes(frame_bgr: np.ndarray, gt_boxes: list) -> np.ndarray:
    """Gambar ground truth box warna putih, label 'GT', biar beda dari box prediksi."""
    for gt in gt_boxes:
        x1, y1, x2, y2 = [int(v) for v in gt["box"]]
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            frame_bgr, "GT", (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame_bgr


def compute_iou(box_a: list, box_b: list) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def _compute_ap(precisions: list, recalls: list) -> float:
    """AP pakai all-point interpolation (cara standar PASCAL VOC / COCO)."""
    if not precisions:
        return 0.0

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def evaluate_against_gt(result, gt_boxes: list, iou_threshold: float = 0.5) -> dict:
    """
    Bandingkan hasil prediksi 1 model ke ground truth (single image, single class 'plat').
    Return precision, recall, AP (all-point interpolation), dan jumlah TP/FP/FN.
    """
    n_gt = len(gt_boxes)

    if result.boxes is None or len(result.boxes) == 0:
        return {"precision": 0.0, "recall": 0.0, "ap": 0.0, "tp": 0, "fp": 0, "fn": n_gt}

    pred_boxes = result.boxes.xyxy.cpu().numpy().tolist()
    pred_confs = result.boxes.conf.cpu().numpy().tolist()

    # urutkan prediksi berdasarkan confidence, tertinggi dulu (dibutuhkan buat hitung AP)
    order = sorted(range(len(pred_confs)), key=lambda i: pred_confs[i], reverse=True)
    pred_boxes = [pred_boxes[i] for i in order]

    gt_matched = [False] * n_gt
    tp_list, fp_list = [], []

    for pbox in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gt_boxes):
            if gt_matched[i]:
                continue
            iou = compute_iou(pbox, gt["box"])
            if iou > best_iou:
                best_iou, best_idx = iou, i

        if best_iou >= iou_threshold and best_idx != -1:
            gt_matched[best_idx] = True
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum, fp_cum = [], []
    running_tp = running_fp = 0
    for tp, fp in zip(tp_list, fp_list):
        running_tp += tp
        running_fp += fp
        tp_cum.append(running_tp)
        fp_cum.append(running_fp)

    precisions = [tp / (tp + fp) if (tp + fp) > 0 else 0.0 for tp, fp in zip(tp_cum, fp_cum)]
    recalls = [tp / n_gt if n_gt > 0 else 0.0 for tp in tp_cum]
    ap = _compute_ap(precisions, recalls)

    final_tp = tp_cum[-1] if tp_cum else 0
    final_fp = fp_cum[-1] if fp_cum else 0
    final_fn = n_gt - final_tp
    final_precision = final_tp / (final_tp + final_fp) if (final_tp + final_fp) > 0 else 0.0
    final_recall = final_tp / n_gt if n_gt > 0 else 0.0

    return {
        "precision": final_precision,
        "recall": final_recall,
        "ap": ap,
        "tp": final_tp,
        "fp": final_fp,
        "fn": final_fn,
    }


def open_video_writer(output_path: str, fps: float, frame_size: tuple):
    """Buat VideoWriter pakai codec mp4v (gak bergantung H.264/OpenH264 yang sering gak ada di server)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if writer.isOpened():
        return writer
    return None