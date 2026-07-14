import os
import shutil
import subprocess
import tempfile
import time

import cv2
import streamlit as st

from config import MODEL_REGISTRY
from model_loader import load_model
from utils import convert_uploaded_image, draw_detections, open_video_writer, summarize_result

st.set_page_config(
    page_title="Perbandingan YOLOv12n + CBAM - Deteksi Plat Nomor",
    layout="wide",
)

st.title("🔍 Perbandingan Model YOLOv12n + CBAM (Ablation Study)")
st.markdown(
    "Bandingkan performa **baseline** vs varian **CBAM** (channel-only, spatial-only, full) "
    "di berbagai checkpoint epoch (20/35/50) untuk deteksi plat nomor kendaraan."
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Pengaturan")

    st.subheader("Pilih model")
    selected_labels = st.multiselect(
        "Model & checkpoint yang mau dibandingkan",
        options=list(MODEL_REGISTRY.keys()),
        default=[list(MODEL_REGISTRY.keys())[0]],
        help="Bisa pilih lebih dari satu. Makin banyak dipilih, makin lama prosesnya.",
    )

    conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.45, 0.05)

    st.divider()
    st.subheader("Input")
    input_type = st.radio("Jenis input", ["Gambar", "Video"], horizontal=True)

    if input_type == "Video":
        use_tracker = st.checkbox("Aktifkan tracking (ByteTrack)", value=True)
        max_frames = st.number_input(
            "Batas jumlah frame diproses",
            min_value=10, max_value=1000, value=150, step=10,
            help="Streamlit Cloud punya batas resource. Batasi frame biar gak timeout.",
        )
    else:
        use_tracker = False
        max_frames = None

    st.divider()
    st.caption(
        "⚠️ Streamlit Community Cloud punya RAM terbatas (~1GB). "
        "Kalau pilih banyak model + video panjang, proses bisa lambat/gagal. "
        "Disarankan mulai dari 2-3 model dulu."
    )

if not selected_labels:
    st.warning("Pilih minimal 1 model di sidebar dulu ya.")
    st.stop()


def _run_predict(model, image_bgr, tracker: bool):
    if tracker:
        return model.track(
            image_bgr, conf=conf_threshold, iou=iou_threshold,
            persist=True, verbose=False,
        )[0]
    return model.predict(
        image_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False,
    )[0]


# ==================== GAMBAR ====================
if input_type == "Gambar":
    uploaded = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

    if uploaded:
        image_bgr = convert_uploaded_image(uploaded)
        st.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            caption="Gambar asli", use_container_width=True,
        )

        if st.button("🚀 Jalankan Deteksi", type="primary"):
            summary_rows = []
            n_cols = min(len(selected_labels), 3)
            cols = st.columns(n_cols)

            for i, label in enumerate(selected_labels):
                repo_id = MODEL_REGISTRY[label]
                col = cols[i % n_cols]

                with col:
                    st.subheader(label)
                    try:
                        with st.spinner("Memuat model & inference..."):
                            model = load_model(repo_id)
                            t0 = time.perf_counter()
                            result = _run_predict(model, image_bgr, tracker=False)
                            infer_ms = (time.perf_counter() - t0) * 1000
                    except Exception as e:
                        st.error(f"Gagal menjalankan model ini: {e}")
                        continue

                    annotated = draw_detections(image_bgr.copy(), result, label)
                    stats = summarize_result(result)

                    st.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )
                    st.metric("Inference time", f"{infer_ms:.1f} ms")
                    st.metric("Jumlah deteksi", stats["n_det"])
                    st.metric("Confidence rata-rata", f"{stats['avg_conf']:.3f}")

                    summary_rows.append({
                        "Model": label,
                        "Jumlah Deteksi": stats["n_det"],
                        "Confidence Rata-rata": round(stats["avg_conf"], 3),
                        "Confidence Maks": round(stats["max_conf"], 3),
                        "Inference Time (ms)": round(infer_ms, 2),
                    })

            if summary_rows:
                st.divider()
                st.subheader("📊 Tabel Perbandingan")
                st.dataframe(summary_rows, use_container_width=True, hide_index=True)


# ==================== VIDEO ====================
elif input_type == "Video":
    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

    if uploaded:
        video_bytes = uploaded.getvalue()
        st.video(video_bytes)

        if st.button("🚀 Proses Video", type="primary"):
            _, ext = os.path.splitext(uploaded.name or "")
            suffix = ext.lower() if ext.lower() in [".mp4", ".avi", ".mov"] else ".mp4"

            tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_in.write(video_bytes)
            tmp_in.close()

            summary_rows = []

            for label in selected_labels:
                repo_id = MODEL_REGISTRY[label]
                st.subheader(label)

                try:
                    with st.spinner(f"Memuat model {label}..."):
                        model = load_model(repo_id)
                except Exception as e:
                    st.error(f"Gagal memuat model ini: {e}")
                    continue

                cap = cv2.VideoCapture(tmp_in.name)
                if not cap.isOpened():
                    st.error("Tidak bisa membuka file video.")
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames_to_process = min(total_frames, max_frames) if total_frames > 0 else max_frames

                tmp_out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                writer = open_video_writer(tmp_out_path, fps, (width, height))
                if writer is None:
                    st.error("Tidak bisa membuat video writer di server ini.")
                    cap.release()
                    continue

                progress = st.progress(0)
                status = st.empty()
                preview = st.empty()

                frame_count = 0
                total_infer_ms = 0.0
                all_confs = []
                unique_track_ids = set()

                try:
                    while frame_count < frames_to_process:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        t0 = time.perf_counter()
                        result = _run_predict(model, frame, tracker=use_tracker)
                        infer_ms = (time.perf_counter() - t0) * 1000
                        total_infer_ms += infer_ms

                        annotated = draw_detections(frame.copy(), result, label)
                        writer.write(annotated)

                        stats = summarize_result(result)
                        all_confs.extend(
                            result.boxes.conf.cpu().numpy().tolist()
                            if result.boxes is not None else []
                        )
                        if use_tracker and getattr(result.boxes, "id", None) is not None:
                            unique_track_ids.update(
                                result.boxes.id.cpu().numpy().astype(int).tolist()
                            )

                        frame_count += 1
                        if frame_count % 10 == 0 or frame_count == frames_to_process:
                            preview.image(
                                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                use_container_width=True,
                            )
                            progress.progress(frame_count / frames_to_process)
                            status.text(f"Frame {frame_count}/{frames_to_process}")
                finally:
                    cap.release()
                    writer.release()

                status.text(f"Selesai — {frame_count} frame diproses.")

                # coba transcode ke H.264 biar bisa diputar langsung di browser
                browser_path = tmp_out_path
                ffmpeg_bin = shutil.which("ffmpeg")
                if ffmpeg_bin:
                    h264_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    cmd = [
                        ffmpeg_bin, "-y", "-i", tmp_out_path,
                        "-vcodec", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", h264_path,
                    ]
                    completed = subprocess.run(cmd, capture_output=True, text=True)
                    if completed.returncode == 0 and os.path.getsize(h264_path) > 0:
                        browser_path = h264_path

                with open(browser_path, "rb") as f:
                    result_bytes = f.read()

                st.video(result_bytes)
                st.download_button(
                    f"Unduh hasil video ({label})",
                    data=result_bytes,
                    file_name=f"hasil_{label.replace(' ', '_')}.mp4",
                    mime="video/mp4",
                )

                avg_infer_ms = total_infer_ms / frame_count if frame_count else 0
                avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

                summary_rows.append({
                    "Model": label,
                    "Frame Diproses": frame_count,
                    "Total Deteksi": len(all_confs),
                    "ID Unik (tracking)": len(unique_track_ids) if use_tracker else "-",
                    "Confidence Rata-rata": round(avg_conf, 3),
                    "Avg Inference Time/Frame (ms)": round(avg_infer_ms, 2),
                })

                if os.path.exists(tmp_out_path):
                    os.unlink(tmp_out_path)
                if browser_path != tmp_out_path and os.path.exists(browser_path):
                    os.unlink(browser_path)

            os.unlink(tmp_in.name)

            if summary_rows:
                st.divider()
                st.subheader("📊 Tabel Perbandingan Semua Model")
                st.dataframe(summary_rows, use_container_width=True, hide_index=True)