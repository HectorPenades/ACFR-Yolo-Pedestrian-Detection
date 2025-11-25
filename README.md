# YOLO12 Pedestrian Evaluation — All-in-One Pipeline  
### Pseudo-GT (YOLO12x) • Predictions (YOLO12s FP16) • AP@0.5 Evaluation • TensorBoard • Plots  
**Version v0.7 — Vorta Edition**

---

## 1. Overview

This repository provides a complete evaluation workflow for pedestrian detection using **Ultralytics YOLOv12** and **OpenVINO**.

The script `eval_all_in_one.py` automates:

- Pseudo-Ground-Truth generation with **YOLO12x**  
- Prediction generation with **YOLO12s FP16 (OpenVINO)**  
- Evaluation of the class **person**:
  - True Positives, False Positives, False Negatives
  - Precision, Recall, F1-score
  - **AP@0.5**
- Optional:
  - TP/FP visualizations
  - TensorBoard metrics logging
  - Metric plots and threshold sweeps

Run the entire pipeline with:

```bash
python3 eval_all_in_one.py
```

---

## 2. Directory Structure

```
ACFR-Yolo-Pedestrian-Detection/
│
├── eval_all_in_one.py
│
└── Images_to_evaluate/
    ├── extracted_images_RGB/
    ├── extracted_images_thermal/
    ├── extracted_images_zoomed/
    │
    ├── pseudo_gt_yolo12x.json
    ├── pred_yolo12s_fp16.json
    │
    ├── eval_viz/
    │   ├── RGB/
    │   ├── THERMAL/
    │   └── ZOOMED/
    │
    ├── plots_metrics/
    │
    └── runs_all_in_one/    # TensorBoard logs
    │
    └── models
```

---

## 3. Requirements

Install required dependencies:

```
pip install ultralytics torch matplotlib tensorboard
```



## 4. Running the Pipeline

### Full pipeline (GT + predictions + evaluation)

```
python3 eval_all_in_one.py
```


---

## 5. Visualizations

Enable visualization saving:

```
python3 eval_all_in_one.py --viz 
```

Limit max visualizations per camera:

```
python3 eval_all_in_one.py --viz --max-viz-per-cam 30
```

Saved to:

```
Images_to_evaluate/eval_viz/<CAMERA>/
```

---

## 6. Plot Generation

Generate plots (precision, recall, F1, AP, heatmaps):

```
python3 eval_all_in_one.py --plot  --gt-conf-list 0.25,0.5  --pred-conf-list 0.1,0.3,0.5
```

Output directory:

```
Images_to_evaluate/plots_metrics/
```

---

## 7. TensorBoard

TensorBoard logs are saved in:

```
Images_to_evaluate/runs_all_in_one/
```

Start TensorBoard:

```
tensorboard --logdir Images_to_evaluate/runs_all_in_one
```

---

## 8. Thresholds

| Parameter | Purpose |
|----------|----------|
| `pred_conf_list` | Filters prediction boxes in `pred_yolo12s_fp16.json` before matching |
| `gt_conf_list` | Filters pseudo-GT detections in `pseudo_gt_yolo12x.json` |

Recommended workflow:

- Generate pseudo-GT with **very low conf**, e.g. `0.001`  
- Evaluate with stricter GT thresholds (`0.3–0.7`)

---


---

## 9. Notes

- Pseudo-GT and predictions are **saved progressively**, allowing interruption/resume  
- Ultralytics downloads `.pt` model files automatically  
- OpenVINO model exports are performed once and reused afterward  

