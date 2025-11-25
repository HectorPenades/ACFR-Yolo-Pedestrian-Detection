import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import urllib.request

# ultralytics (se usa tanto para GT generation como para preds OpenVINO/.pt)
from ultralytics import YOLO

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    raise ImportError("Instala tensorboard: pip install tensorboard") from e

# OpenCV opcional para visualizaciones
try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False

# matplotlib para plots
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# --------------------------
# CONFIG DEFAULTS (ajustables)
# --------------------------
IMAGES_DIR = Path("/media/its/Stephany_120G/ACFR-Yolo-Pedestrian-Detection/Images_to_evaluate").resolve()
CAMERAS: Dict[str, Path] = {
    "RGB": IMAGES_DIR / "extracted_images_RGB",
    "THERMAL": IMAGES_DIR / "extracted_images_thermal",
    "ZOOMED": IMAGES_DIR / "extracted_images_zoomed",
}
CAMERA_LABELS = list(CAMERAS.keys())

# Archivo GT -> cambiar a pseudo_gt_yolo12x.json para usar yolo12x
GT_JSON_PATH = IMAGES_DIR / "pseudo_gt_yolo12x.json"
PRED_JSON_PATH = IMAGES_DIR / "pred_yolo12s_fp16.json"

# Model names (Ultralytics puede descargar .pt automáticamente)
YOLO12X_PT = "model/yolo12x.pt"        # modelo grande (para pseudo-GT)
YOLO12S_PT = "model/yolo12s.pt"        # modelo small (para evaluación)

# Directorios OpenVINO (se crearán en IMAGES_DIR)
YOLO12X_OV_DIR = IMAGES_DIR / "model/yolo12x_openvino_model"
YOLO12S_OV_DIR = IMAGES_DIR / "model/yolo12s_fp16_openvino_model"

DEVICE = "CPU"  # para OpenVINO -> "intel:cpu"
IOU_THRESH = 0.5
DEFAULT_GT_CONF_MIN = 0.5
PRED_CONF_LIST = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7]
CONF_THRES_PRED_SAVE = 0.001
BATCH_SIZE = 64
TB_LOGDIR = IMAGES_DIR / "runs_all_in_one"
PERSON_CLASS_ID = 0

os.makedirs(IMAGES_DIR, exist_ok=True)
# Asegurar que el script usa IMAGES_DIR como directorio de trabajo (coherente con infer_yolo.py)
try:
    os.chdir(IMAGES_DIR)
    print(f"[INFO] Directorio de trabajo cambiado a IMAGES_DIR: {IMAGES_DIR}")
except Exception as e:
    print(f"[WARN] No se pudo cambiar el directorio de trabajo a IMAGES_DIR: {e}")


# --------------------------
# UTILIDADES
# --------------------------
def collect_image_paths() -> List[str]:
    exts = {".jpg", ".jpeg ",".png", ".bmp", ".tif", ".tiff"}
    imgs: List[str] = []
    per_cam_counts: Dict[str, int] = {}
    for cam_label, folder in CAMERAS.items():
        per_cam_counts[cam_label] = 0
        if not folder.exists():
            print(f"[AVISO] Carpeta no existe ({cam_label}): {folder}")
            continue
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                imgs.append(str(p.resolve()))
                per_cam_counts[cam_label] += 1

    imgs = sorted(set(imgs))
    total_found = len(imgs)
    for cam_label, cnt in per_cam_counts.items():
        print(f"[INFO] Imágenes en {cam_label}: {cnt}")

    if total_found == 0:
        print(f"[WARN] No se encontraron imágenes en las carpetas de cámaras. Haciendo fallback: scan recursivo en {IMAGES_DIR}")
        for p in IMAGES_DIR.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                imgs.append(str(p.resolve()))
        imgs = sorted(set(imgs))
        total_found = len(imgs)
        print(f"[INFO] Imágenes encontradas en fallback (IMAGES_DIR): {total_found}")
    else:
        print(f"[INFO] Total imágenes encontradas (todas las cámaras): {total_found}")

    return imgs


def get_camera_label_for_image(img_path: Path) -> str:
    for cam_label, cam_path in CAMERAS.items():
        try:
            if img_path.is_relative_to(cam_path):
                return cam_label
        except Exception:
            pass
    return "UNKNOWN"


def compute_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union = area1 + area2 - inter_area + 1e-9
    return inter_area / union if union > 0 else 0.0


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def visualize_detections(image_path: str,
                         gt_annots: List[Dict[str, Any]],
                         preds_tp: List[Dict[str, Any]],
                         preds_fp: List[Dict[str, Any]],
                         out_path: Path):
    if not _HAVE_CV2:
        return
    p = Path(image_path)
    if not p.exists() or p.stat().st_size == 0:
        print(f"[WARN] visualize_detections: imagen no existe o vacía, omitiendo -> {image_path}")
        return
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] visualize_detections: cv2.imread devolvió None para {image_path} (archivo corrupto o formato no soportado).")
        return

    def draw_box(bbox, color, label=None, thickness=2):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            ((w, h), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 18), (x1 + w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    for ann in gt_annots:
        bbox = ann.get("bbox")
        if bbox:
            draw_box(bbox, (0, 255, 0), "GT")
    for p in preds_tp:
        draw_box(p["bbox"], (255, 0, 0), f"TP:{p['conf']:.2f}")
    for p in preds_fp:
        draw_box(p["bbox"], (0, 0, 255), f"FP:{p['conf']:.2f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# --------------------------
# GENERAR PSEUDO-GT (solo personas)
# --------------------------
def generate_gt(gt_model: str,
                out_path: Path,
                images: List[str],
                gt_conf_min: float = DEFAULT_GT_CONF_MIN,
                person_class_id: int = PERSON_CLASS_ID,
                force: bool = False):
    if out_path.exists() and not force:
        print(f"[INFO] GT ya existe en {out_path}, usar --force-gt para regenerar.")
        return

    if not images:
        raise RuntimeError("No hay imágenes para generar GT.")

    print(f"[INFO] Generando GT con modelo: {gt_model} -> guardando en {out_path}")
    model = YOLO(gt_model)
    store = {"images": {}, "annotations": {}, "class_names": {str(person_class_id): "person"}}
    total = len(images)
    for start in range(0, total, BATCH_SIZE):
        batch = images[start:start+BATCH_SIZE]
        print(f"[GT] Inferencia batch {start+1}-{start+len(batch)} / {total}")
        results = model(batch, imgsz=640, conf=gt_conf_min, verbose=False)
        for img_path, res in zip(batch, results):
            camera = get_camera_label_for_image(Path(img_path))
            store["images"][img_path] = {"camera": camera}
            ann_list = []
            # recorrer cajas; filtramos por clase persona (si el modelo reporta cls)
            for b in res.boxes:
                cls_id = int(b.cls[0]) if hasattr(b, "cls") else None
                conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                if cls_id is None:
                    # si no hay cls, asumimos persona si se pidió (pero lo normal es que haya cls)
                    continue
                if cls_id != person_class_id:
                    continue
                if conf < gt_conf_min:
                    continue
                ann_list.append({"cls": int(cls_id), "conf": float(conf), "bbox": [float(x1), float(y1), float(x2), float(y2)]})
            store["annotations"][img_path] = ann_list
    with open(out_path, "w") as f:
        json.dump(store, f, indent=2)
    print(f"[OK] GT guardado en {out_path}")


# --------------------------
# GENERAR PREDICCIONES (YOLO12s)
# --------------------------
def ensure_openvino_model(pt_name: str, ov_dir: Path):
    if ov_dir.exists() and any(ov_dir.glob("*.xml")):
        return
    print("[INFO] Exportando a OpenVINO (FP16) desde", pt_name)
    base = YOLO(pt_name)
    base.export(format="openvino", dynamic=True, half=True, imgsz=640)
    print("[OK] Exportado OpenVINO en", ov_dir)


def generate_predictions(pred_model: str,
                         out_path: Path,
                         image_list_from_gt: List[str],
                         device: str = DEVICE,
                         conf_thres_save: float = CONF_THRES_PRED_SAVE,
                         force: bool = False):
    if out_path.exists() and not force:
        print(f"[INFO] Predicciones ya existen en {out_path}, usar --force-pred para regenerar.")
        return

    # si pred_model apunta a un directorio OpenVINO, YOLO lo aceptará
    print("[INFO] Cargando modelo de predicción:", pred_model)
    det = YOLO(str(pred_model), task="detect")
    pred_store = {}
    total = len(image_list_from_gt)
    start_time = time.time()
    for start in range(0, total, BATCH_SIZE):
        batch = image_list_from_gt[start:start+BATCH_SIZE]
        print(f"[PRED] Batch {start+1}-{start+len(batch)} / {total}")
        results = det(batch, device=f"intel:{device.lower()}", imgsz=640, conf=conf_thres_save, verbose=False)
        for img_path, res in zip(batch, results):
            arr = []
            for b in res.boxes:
                conf = float(b.conf[0])
                if conf < conf_thres_save:
                    continue
                cls_id = int(b.cls[0])
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                arr.append({"cls": cls_id, "conf": conf, "bbox": [float(x1), float(y1), float(x2), float(y2)]})
            pred_store[img_path] = arr
    with open(out_path, "w") as f:
        json.dump(pred_store, f, indent=2)
    elapsed = time.time() - start_time
    print(f"[OK] Predicciones guardadas en {out_path} (tiempo {elapsed/60:.2f} min)")


# --------------------------
# EVALUACIÓN - enfoque PERSONA (como tus scripts)
# --------------------------
def resolve_image_path(img_path: str) -> str:
    """
    Normaliza la ruta desde el JSON: conservar sólo las dos últimas componentes
    (p.ej. 'extracted_images_thermal/1758856533.023745.jpg') y resolver como
    IMAGES_DIR / <carpeta> / <archivo>. Si no existe, hacer fallback por nombre.
    """
    p = Path(img_path)
    try:
        if p.exists() and p.stat().st_size > 0:
            return str(p)
    except Exception:
        pass

    parts = [part for part in p.parts if part not in ("/", "")]
    if len(parts) >= 2:
        last_two = Path(parts[-2]) / parts[-1]
    else:
        last_two = Path(parts[-1]) if parts else Path(img_path)

    candidate = IMAGES_DIR / last_two
    try:
        if candidate.exists() and candidate.stat().st_size > 0:
            print(f"[INFO] resolve_image_path: {img_path} -> {candidate}")
            return str(candidate.resolve())
    except Exception:
        pass

    # fallback: buscar por nombre en IMAGES_DIR
    name = p.name
    for q in IMAGES_DIR.rglob(name):
        try:
            if q.is_file() and q.stat().st_size > 0:
                print(f"[INFO] resolve_image_path fallback: {img_path} -> {q}")
                return str(q.resolve())
        except Exception:
            continue

    return str(p)


def normalize_json_paths(images_dict: Dict[str, Any], annotations_dict: Dict[str, Any]) -> (Dict[str, Any], Dict[str, Any]):
    images_norm: Dict[str, Any] = {}
    annotations_norm: Dict[str, Any] = {}
    remapped = 0
    for orig_path, img_info in images_dict.items():
        resolved = resolve_image_path(orig_path)
        if resolved != orig_path:
            remapped += 1
        images_norm[resolved] = img_info
        annotations_norm[resolved] = annotations_dict.get(orig_path, [])
    if remapped:
        print(f"[INFO] normalize_json_paths: remapeadas {remapped} rutas -> ahora usan prefijo IMAGES_DIR")
    return images_norm, annotations_norm


def evaluate_person(gt_json_path: Path,
                    pred_json_path: Path,
                    pred_conf_list: List[float],
                    gt_conf_min: float,
                    viz: bool = False,
                    viz_outdir: Optional[Path] = None,
                    viz_pred_conf: Optional[float] = None,
                    viz_gt_conf: Optional[float] = None,
                    max_viz_per_cam: int = 50):
    if not gt_json_path.exists() or not pred_json_path.exists():
        raise FileNotFoundError("GT o pred no encontrado.")

    with open(gt_json_path, "r") as f:
        gt_data = json.load(f)
    with open(pred_json_path, "r") as f:
        pred_data = json.load(f)

    # Normalizar rutas GT
    images_info = gt_data.get("images", {})
    annotations = gt_data.get("annotations", {})
    images_info, annotations = normalize_json_paths(images_info, annotations)

    # Normalizar rutas PRED
    pred_data_norm: Dict[str, List[Dict[str, Any]]] = {}
    remapped_pred = 0
    for orig_p, annots in pred_data.items():
        resolved = resolve_image_path(orig_p)
        if resolved != orig_p:
            remapped_pred += 1
        pred_data_norm[resolved] = annots
    if remapped_pred:
        print(f"[INFO] normalize_json_paths: remapeadas {remapped_pred} rutas en pred_data")
    pred_data = pred_data_norm

    # ahora images_info / annotations / pred_data usan rutas resueltas (con ACFR-...)
    img_to_cam = {p: info.get("camera", "UNKNOWN") for p, info in images_info.items()}

    writer = SummaryWriter(log_dir=str(TB_LOGDIR))

    if viz and viz_outdir is None:
        viz_outdir = IMAGES_DIR / "eval_viz"
    saved_viz_cam = {cam: 0 for cam in CAMERA_LABELS}

    for pred_conf_min in pred_conf_list:
        step = int(pred_conf_min * 100)
        preds_person = []
        n_gt_person_global = 0
        total_tps = 0
        total_fps = 0
        n_pred_person_global = 0

        for img_path in sorted(images_info.keys()):
            preds_raw = pred_data.get(img_path, [])
            anns_raw = annotations.get(img_path, [])
            camera = img_to_cam.get(img_path, "UNKNOWN")

            anns_gt = [a for a in anns_raw if int(a.get("cls", -1)) == PERSON_CLASS_ID and float(a.get("conf", 0.0)) >= gt_conf_min]
            preds = [p for p in preds_raw if int(p.get("cls", -1)) == PERSON_CLASS_ID and float(p.get("conf", 0.0)) >= pred_conf_min]

            if not anns_gt and not preds:
                continue

            n_gt_person_global += len(anns_gt)
            n_pred_person_global += len(preds)

            gt_list = [{"bbox": ann["bbox"], "used": False} for ann in anns_gt]
            preds_sorted = sorted(preds, key=lambda x: float(x.get("conf", 0.0)), reverse=True)

            preds_tp_list = []
            preds_fp_list = []

            for pred in preds_sorted:
                bbox_pred = pred["bbox"]
                conf = float(pred.get("conf", 0.0))
                best_iou = 0.0
                best_gt_idx = -1
                for i_gt, gt_obj in enumerate(gt_list):
                    if gt_obj["used"]:
                        continue
                    iou = compute_iou(bbox_pred, gt_obj["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i_gt
                if best_iou >= IOU_THRESH and best_gt_idx >= 0:
                    gt_list[best_gt_idx]["used"] = True
                    preds_person.append({"conf": conf, "tp": 1, "fp": 0, "camera": camera})
                    preds_tp_list.append({"bbox": bbox_pred, "conf": conf})
                    total_tps += 1
                else:
                    preds_person.append({"conf": conf, "tp": 0, "fp": 1, "camera": camera})
                    preds_fp_list.append({"bbox": bbox_pred, "conf": conf})
                    total_fps += 1

            # visualización
            if viz and (preds_fp_list or preds_tp_list):
                if viz_pred_conf is not None and abs(pred_conf_min - float(viz_pred_conf)) > 1e-9:
                    pass
                else:
                    # filtrar GT para viz si se pidió
                    if viz_gt_conf is not None:
                        anns_gt_viz = [a for a in anns_raw if int(a.get("cls", -1)) == PERSON_CLASS_ID and float(a.get("conf", 0.0)) >= viz_gt_conf]
                    else:
                        anns_gt_viz = anns_gt
                    if saved_viz_cam.get(camera, 0) < max_viz_per_cam:
                        out_name = f"viz_{camera}_{saved_viz_cam[camera]+1:03d}_{Path(img_path).name}"
                        out_path = Path(viz_outdir) / camera / out_name
                        visualize_detections(img_path, anns_gt_viz, preds_tp_list, preds_fp_list, out_path)
                        saved_viz_cam[camera] += 1

        # Tras procesar todas las imágenes para este umbral, calcular FN y mostrar resumen
        total_tps_global = total_tps
        total_fps_global = total_fps
        total_fn_global = max(0, n_gt_person_global - total_tps_global)

        print(f"[SUMMARY] conf_pred >= {pred_conf_min} -> TP={total_tps_global}, FP={total_fps_global}, FN={total_fn_global}, GT={n_gt_person_global}, PRED={n_pred_person_global}")
        writer.add_scalar("person/TP_global", total_tps_global, step)
        writer.add_scalar("person/FP_global", total_fps_global, step)
        writer.add_scalar("person/FN_global", total_fn_global, step)

        # === métricas globales: precision / recall / f1 (basadas en totales TP/FP/FN) ===
        # calcular métricas globales una sola vez (usar los contadores acumulados)
        total_tp = total_tps_global
        total_fp = total_fps_global
        total_fn = total_fn_global

        if (total_tp + total_fp) > 0:
            precision_global = total_tp / (total_tp + total_fp)
        else:
            precision_global = 0.0
        if n_gt_person_global > 0:
            recall_global = total_tp / (n_gt_person_global + 1e-9)
        else:
            recall_global = 0.0
        if (precision_global + recall_global) > 0:
            f1_global = 2 * (precision_global * recall_global) / (precision_global + recall_global)
        else:
            f1_global = 0.0

        # imprimir y subir a TensorBoard
        print(f"[METRICS GLOBAL] precision={precision_global:.4f}, recall={recall_global:.4f}, f1={f1_global:.4f}")
        writer.add_scalar("person/precision_global", precision_global, step)
        writer.add_scalar("person/recall_global", recall_global, step)
        writer.add_scalar("person/f1_global", f1_global, step)

        # AP global
        if n_gt_person_global == 0 or not preds_person:
            print("[AVISO] No hay GT o pred para calcular AP.")
            continue
        preds_sorted_global = sorted(preds_person, key=lambda x: x["conf"], reverse=True)
        tps_arr = np.array([p["tp"] for p in preds_sorted_global])
        fps_arr = np.array([p["fp"] for p in preds_sorted_global])
        tps_cum = np.cumsum(tps_arr)
        fps_cum = np.cumsum(fps_arr)
        recalls = tps_cum / (n_gt_person_global + 1e-9)
        precisions = tps_cum / (tps_cum + fps_cum + 1e-9)
        ap_person_global = compute_ap(recalls, precisions)
        writer.add_scalar("person/AP50_global", ap_person_global, step)
        print(f"[RESULT] AP@0.5 global = {ap_person_global:.4f}")

    writer.close()
    print(f"[OK] Evaluación subida a TensorBoard en {TB_LOGDIR}")


# --------------------------
# PLOTEO SIMPLE
# --------------------------
def compute_metrics_matrix(gt_data: Dict[str, Any],
                           pred_data: Dict[str, Any],
                           gt_conf_list: List[float],
                           pred_conf_list: List[float]) -> Dict[str, np.ndarray]:
    images_info = gt_data.get("images", {})
    annotations = gt_data.get("annotations", {})
    m = len(gt_conf_list); n = len(pred_conf_list)
    metrics = {
        "precision": np.zeros((m, n), dtype=float),
        "recall": np.zeros((m, n), dtype=float),
        "f1": np.zeros((m, n), dtype=float),
        "ap": np.zeros((m, n), dtype=float),
    }
    for i, gt_conf in enumerate(gt_conf_list):
        for j, pred_conf in enumerate(pred_conf_list):
            res = compute_metrics_single(images_info, annotations, pred_data, gt_conf, pred_conf)
            metrics["precision"][i, j] = res["precision"]
            metrics["recall"][i, j] = res["recall"]
            metrics["f1"][i, j] = res["f1"]
            metrics["ap"][i, j] = res["ap"]
    return metrics


def compute_metrics_matrix_by_camera(gt_data: Dict[str, Any],
                                     pred_data: Dict[str, Any],
                                     gt_conf_list: List[float],
                                     pred_conf_list: List[float],
                                     camera_label: str) -> Dict[str, np.ndarray]:
    """
    Filtra GT por camera_label y llama a compute_metrics_matrix sobre ese subconjunto.
    """
    images_info = gt_data.get("images", {})
    annotations = gt_data.get("annotations", {})

    # Filtrar imágenes pertenecientes a la cámara
    images_info_cam = {p: info for p, info in images_info.items() if info.get("camera") == camera_label}
    # Filtrar anotaciones por las imágenes seleccionadas
    annotations_cam = {p: annotations.get(p, []) for p in images_info_cam.keys()}

    gt_sub = {
        "images": images_info_cam,
        "annotations": annotations_cam,
    }

    return compute_metrics_matrix(gt_sub, pred_data, gt_conf_list, pred_conf_list)


def compute_metrics_single(images_info: Dict[str, Any],
                           annotations: Dict[str, Any],
                           pred_data: Dict[str, Any],
                           gt_conf: float,
                           pred_conf: float) -> Dict[str, float]:
    preds_person = []
    n_gt = 0
    for img_path, info in images_info.items():
        anns_raw = annotations.get(img_path, [])
        preds_raw = pred_data.get(img_path, [])
        anns = [a for a in anns_raw if int(a.get("cls", -1)) == PERSON_CLASS_ID and float(a.get("conf", 0.0)) >= gt_conf]
        preds = [p for p in preds_raw if int(p.get("cls", -1)) == PERSON_CLASS_ID and float(p.get("conf", 0.0)) >= pred_conf]
        n_gt += len(anns)
        gt_list = [{"bbox": a["bbox"], "used": False} for a in anns]
        preds_sorted = sorted(preds, key=lambda x: float(x.get("conf", 0.0)), reverse=True)
        for p in preds_sorted:
            conf = float(p.get("conf", 0.0))
            bbox = p["bbox"]
            best_iou = 0.0
            best_idx = -1
            for i, g in enumerate(gt_list):
                if g["used"]:
                    continue
                iou = compute_iou(bbox, g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= IOU_THRESH and best_idx >= 0:
                gt_list[best_idx]["used"] = True
                preds_person.append({"conf": conf, "tp": 1, "fp": 0})
            else:
                preds_person.append({"conf": conf, "tp": 0, "fp": 1})
    total_tp = sum([p["tp"] for p in preds_person])
    total_fp = sum([p["fp"] for p in preds_person])
    total_fn = max(0, n_gt - total_tp)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (n_gt + 1e-9) if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    if n_gt == 0 or not preds_person:
        ap = 0.0
    else:
        preds_sorted_global = sorted(preds_person, key=lambda x: x["conf"], reverse=True)
        tps_arr = np.array([p["tp"] for p in preds_sorted_global])
        fps_arr = np.array([p["fp"] for p in preds_sorted_global])
        tps_cum = np.cumsum(tps_arr)
        fps_cum = np.cumsum(fps_arr)
        recalls = tps_cum / (n_gt + 1e-9)
        precisions = tps_cum / (tps_cum + fps_cum + 1e-9)
        ap = compute_ap(recalls, precisions)
    return {"TP": total_tp, "FP": total_fp, "FN": total_fn, "GT": n_gt, "PRED": len(preds_person), "precision": precision, "recall": recall, "f1": f1, "ap": ap}


def plot_metrics(metrics: Dict[str, np.ndarray],
                 gt_conf_list: List[float],
                 pred_conf_list: List[float],
                 outdir: Path):
    if plt is None:
        print("[WARN] matplotlib no disponible, se omiten plots.")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    x = pred_conf_list
    for metric_name in ["precision", "recall", "f1", "ap"]:
        plt.figure(figsize=(8,5))
        for i, gt_conf in enumerate(gt_conf_list):
            y = metrics[metric_name][i, :]
            plt.plot(x, y, marker='o', label=f"GT={gt_conf}")
        plt.xlabel("pred_conf")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs pred_conf")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{metric_name}_lines.png", dpi=150)
        plt.close()


# --------------------------
# MAIN CLI
# --------------------------
def maybe_download_file(dest: Path, url: Optional[str], force: bool = False):
    """
    Descarga simple con urllib si dest no existe y url está provista.
    """
    if dest.exists() and not force:
        print(f"[INFO] Archivo ya existe: {dest}")
        return
    if not url:
        print(f"[INFO] No se proporcionó URL para descargar {dest.name}. Descárgalo manualmente y colócalo en {dest}")
        print(f"  Ejemplo (wget): wget <URL_DEL_MODELO> -O {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Descargando {url} -> {dest} ...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        print(f"[OK] Descargado: {dest}")
    except Exception as e:
        print(f"[ERROR] Falló la descarga: {e}. Descarga manualmente con wget/curl.")


def main():
    parser = argparse.ArgumentParser(description="Script ALL-IN-ONE: generar GT (yolo12x), generar pred (yolo12s), evaluar persona, viz y plots.")
    parser.add_argument("--gt-model", type=str, default=YOLO12X_PT, help="Modelo para generar GT (por defecto yolo12x.pt).")
    parser.add_argument("--force-gt", action="store_true", help="Forzar regeneración del GT aunque exista.")
    parser.add_argument("--gt-conf-min", type=float, default=DEFAULT_GT_CONF_MIN, help="Umbral mínimo para guardar detecciones en el GT (al generar).")
    parser.add_argument("--skip-pred-gen", action="store_true", help="No generar predicciones (usar pred JSON si existe).")
    parser.add_argument("--force-pred", action="store_true", help="Forzar regeneración de predicciones aunque existan.")
    parser.add_argument("--pred-model", type=str, default=YOLO12S_PT, help="Modelo para predicciones (por defecto yolo12s.pt).")
    parser.add_argument("--pred-conf-list", type=str, default=",".join(map(str, PRED_CONF_LIST)), help="CSV lista umbrales pred (ej: 0.1,0.3,0.5).")
    parser.add_argument("--max-viz-per-cam", type=int, default=50, help="Máximo de visualizaciones guardadas por cámara.")
    parser.add_argument("--viz", action="store_true", help="Guardar visualizaciones TP/FP.")
    parser.add_argument("--viz-outdir", type=str, default=str(IMAGES_DIR / "eval_viz"), help="Directorio visualizaciones.")
    parser.add_argument("--viz-pred-conf", type=float, default=None, help="Umbral pred para seleccionar iteración de visualización.")
    parser.add_argument("--viz-gt-conf", type=float, default=None, help="Umbral GT para filtrar visualizaciones.")
    parser.add_argument("--plot", action="store_true", help="Generar plots de métricas vs umbrales.")
    parser.add_argument("--plots-outdir", type=str, default=str(IMAGES_DIR / "plots_metrics"), help="Directorio donde guardar plots.")
    parser.add_argument("--gt-conf-list", type=str, default=None, help="CSV lista de umbrales GT para plots/evaluación (ej: 0.25,0.5).")
    parser.add_argument("--download-model", nargs="?", const=True, default=False, help="Intentar descargar modelos si faltan (usa --download-pred-model-url / --download-gt-model-url).")
    parser.add_argument("--download-pred-model-url", type=str, default=None, help="URL para descargar yolo12s.pt (opcional).")
    parser.add_argument("--download-gt-model-url", type=str, default=None, help="URL para descargar yolo12x.pt (opcional).")
    args = parser.parse_args()

    # Mostrar resumen claro de flags por defecto/efectivos
    print(f"[FLAGS] viz={args.viz}  (guardar visualizaciones)")
    if args.viz:
        print(f"[FLAGS]  -> max_viz_per_cam={args.max_viz_per_cam}, viz_outdir={args.viz_outdir}, viz_pred_conf={args.viz_pred_conf}, viz_gt_conf={args.viz_gt_conf}")
        print(f"[INFO] Las visualizaciones se guardarán en: {args.viz_outdir} (por cámara). Límite por cámara: {args.max_viz_per_cam} imágenes.")
    print(f"[FLAGS] plot={args.plot}  (generar plots)")
    if args.plot:
        print(f"[FLAGS]  -> plots_outdir={args.plots_outdir}, gt_conf_list={args.gt_conf_list or 'default'}, pred_conf_list={args.pred_conf_list}")
    print(f"[FLAGS] download_model={args.download_model}  (intentar descargar modelos)")
    if args.download_model:
        print(f"[FLAGS]  -> download-pred-model-url={args.download_pred_model_url}, download-gt-model-url={args.download_gt_model_url}")
        if not args.download_pred_model_url and not args.download_gt_model_url:
            print("[FLAGS]  -> No se proporcionaron URLs. Si usas nombres como 'yolo12x.pt' o 'yolo12s.pt', ultralytics puede descargar automáticamente los .pt al cargarlos.")

    # imágenes a procesar
    images = collect_image_paths()
    if not images:
        print("[ERROR] No se encontraron imágenes en las carpetas de cámaras.")
        return

    # intentar descargar si se pidió
    if args.download_model:
        maybe_download_file(Path(YOLO12S_PT), args.download_pred_model_url)
        maybe_download_file(Path(YOLO12X_PT), args.download_gt_model_url)

    # Mensajes claros si ya existen los JSON y no se van a generar
    if GT_JSON_PATH.exists() and not args.force_gt:
        print(f"[INFO] Detectado pseudo-GT JSON existente: {GT_JSON_PATH}. No se regenerará (usa --force-gt para forzar).")
    if PRED_JSON_PATH.exists() and not args.force_pred:
        print(f"[INFO] Detectado pred JSON existente: {PRED_JSON_PATH}. No se regenerará (usa --force-pred para forzar).")

    # --- 1) GENERAR/COMPROBAR PSEUDO-GT (yolo12x) ---
    if not GT_JSON_PATH.exists() or args.force_gt:
        print("[INFO] pseudo-GT no existe o forzado -> generando con yolo12x")
        ensure_openvino_model(args.gt_model, YOLO12X_OV_DIR)
        generate_gt(args.gt_model, GT_JSON_PATH, images, gt_conf_min=args.gt_conf_min, force=args.force_gt)
    else:
        print(f"[INFO] Usando GT existente: {GT_JSON_PATH}")

    # --- 2) GENERAR/COMPROBAR PREDICCIONES (yolo12s) ---
    if not args.skip_pred_gen:
        if not PRED_JSON_PATH.exists() or args.force_pred:
            print("[INFO] Pred JSON no existe o forzado -> generando con yolo12s")
            ensure_openvino_model(args.pred_model, YOLO12S_OV_DIR)
            pred_model_to_use = str(YOLO12S_OV_DIR)
            generate_predictions(pred_model_to_use, PRED_JSON_PATH, sorted({p for p in images if str(p) != ""}), device=DEVICE, conf_thres_save=CONF_THRES_PRED_SAVE, force=args.force_pred)
        else:
            print(f"[INFO] Usando pred JSON existente: {PRED_JSON_PATH}")
    else:
        print("[INFO] Omitida generación de predicciones (--skip-pred-gen).")

    # --- 3) EVALUACIÓN y VISUALIZACIÓN ---
    # parsear listas de umbrales
    pred_conf_list = [float(x) for x in args.pred_conf_list.split(",") if x.strip() != ""]
    # indicar si el usuario proporcionó explícitamente --gt-conf-list
    gt_conf_list_provided = bool(args.gt_conf_list)
    if args.gt_conf_list:
        gt_conf_list = [float(x) for x in args.gt_conf_list.split(",") if x.strip() != ""]
    else:
        gt_conf_list = [args.gt_conf_min]

    # Si el usuario pidió plots y no pasó gt_conf_list explícita, usar lista por defecto
    if args.plot and not gt_conf_list_provided:
        # usar varios GTs por defecto para que las gráficas muestren múltiples curvas
        gt_conf_list = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7]

    print(f"[INFO] Umbrales GT para FILTRADO de anotaciones al evaluar: {gt_conf_list}")
    print(f"[INFO] Umbrales PRED para FILTRADO de predicciones al evaluar: {pred_conf_list}")
    print("[INFO] Nota: los umbrales 'pred_conf_list' se aplican a las predicciones (fichero pred JSON).")
    print("[INFO]       El umbral 'gt_conf' (GT) se usa para filtrar las anotaciones del pseudo-GT cuando computes métricas.")
    print("[INFO]       Si generaste el pseudo-GT con un umbral distinto, usa --gt-conf-list o regenera con --force-gt y --gt-gen-conf si fuera necesario.")

    # Para evaluación principal (iterar sobre pred_conf_list y filtrar GT con args.gt_conf_min)
    evaluate_person(
        GT_JSON_PATH,
        PRED_JSON_PATH,
        pred_conf_list,
        gt_conf_min=args.gt_conf_min,
        viz=args.viz,
        viz_outdir=Path(args.viz_outdir),
        viz_pred_conf=args.viz_pred_conf,
        viz_gt_conf=args.viz_gt_conf,
        max_viz_per_cam=args.max_viz_per_cam,
    )

    # --- 4) Plots: barrido GT confs x pred confs si se pide ---
    if args.plot:
        with open(GT_JSON_PATH, "r") as f:
            gt_data = json.load(f)
        with open(PRED_JSON_PATH, "r") as f:
            pred_data = json.load(f)

        print(f"[INFO] Generando plots para GT confs: {gt_conf_list} y pred confs: {pred_conf_list}")
        metrics = compute_metrics_matrix(gt_data, pred_data, gt_conf_list, pred_conf_list)

        # Directorio principal de plots
        plots_dir = Path(args.plots_outdir)
        plot_metrics(metrics, gt_conf_list, pred_conf_list, plots_dir)
        print(f"[OK] Plots guardados en: {plots_dir}")

        # Resumen global y subir a TensorBoard: mAP (mean AP sobre pred_conf_list), y curvas por GT
        writer = SummaryWriter(log_dir=str(TB_LOGDIR))
        for i, gt_conf in enumerate(gt_conf_list):
            ap_curve = metrics["ap"][i, :]
            prec_curve = metrics["precision"][i, :]
            rec_curve = metrics["recall"][i, :]
            f1_curve = metrics["f1"][i, :]
            # mAP: promedio simple de AP sobre pred_conf_list (una métrica resumen)
            mAP = float(np.mean(ap_curve)) if ap_curve.size > 0 else 0.0
            print(f"[SUMMARY] GT={gt_conf} -> mAP(mean over pred_confs) = {mAP:.4f}")
            writer.add_scalar(f"person/mAP_gt_{gt_conf}", mAP, 0)
            # subir curvas por threshold a TensorBoard
            for j, pred_conf in enumerate(pred_conf_list):
                step = int(pred_conf * 100)
                writer.add_scalar(f"person/precision_gt{gt_conf}", float(prec_curve[j]), step)
                writer.add_scalar(f"person/recall_gt{gt_conf}", float(rec_curve[j]), step)
                writer.add_scalar(f"person/f1_gt{gt_conf}", float(f1_curve[j]), step)
                writer.add_scalar(f"person/AP_gt{gt_conf}", float(ap_curve[j]), step)

        # Plots y métricas por cámara
        for cam in CAMERA_LABELS:
            outdir_cam = plots_dir / "by_camera" / cam
            print(f"[INFO] Generando plots para cámara: {cam} -> {outdir_cam}")
            metrics_cam = compute_metrics_matrix_by_camera(gt_data, pred_data, gt_conf_list, pred_conf_list, cam)
            plot_metrics(metrics_cam, gt_conf_list, pred_conf_list, outdir_cam)
            # resumen mAP por cámara
            for i, gt_conf in enumerate(gt_conf_list):
                ap_curve_cam = metrics_cam["ap"][i, :]
                mAP_cam = float(np.mean(ap_curve_cam)) if ap_curve_cam.size > 0 else 0.0
                print(f"  [CAM {cam}] GT={gt_conf} -> mAP = {mAP_cam:.4f}")
                writer.add_scalar(f"person/mAP_{cam}_gt_{gt_conf}", mAP_cam, 0)
        writer.close()
        print(f"[OK] Plots por cámara guardados en: {plots_dir / 'by_camera'}")


if __name__ == "__main__":
    main()
