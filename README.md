# Evaluación Yolo (eval_all_in_one.py) — Resumen rápido / Quick summary

ES — Propósito
- `eval_all_in_one.py` automatiza:
  - Generación de pseudo-GT usando yolo12x (`pseudo_gt_yolo12x.json`).
  - Generación de predicciones usando yolo12s (`pred_yolo12s_fp16.json`).
  - Evaluación de la clase "person": TP / FP / FN, precision, recall, F1, AP@0.5.
  - Guardado de visualizaciones (TP/FP), métricas a TensorBoard y plots.

EN — Purpose
- `eval_all_in_one.py` automates:
  - Generating pseudo-GT with yolo12x (`pseudo_gt_yolo12x.json`).
  - Generating predictions with yolo12s (`pred_yolo12s_fp16.json`).
  - Evaluating the "person" class: TP / FP / FN, precision, recall, F1, AP@0.5.
  - Saving visualizations (TP/FP), logging scalars to TensorBoard and creating plots.

Rutas por defecto / Default paths
- Proyecto / Project:
  /media/its/Stephany_120G/ACFR-Yolo-Pedestrian-Detection/
- Carpeta de imágenes / Images root:
  /media/its/Stephany_120G/ACFR-Yolo-Pedestrian-Detection/Images_to_evaluate/
  - Subcarpetas esperadas por cámara / expected subfolders:
    - extracted_images_RGB/
    - extracted_images_thermal/
    - extracted_images_zoomed/
- Archivos JSON (generados) / Generated JSONs:
  - pseudo-GT: `.../Images_to_evaluate/pseudo_gt_yolo12x.json`
  - predicciones: `.../Images_to_evaluate/pred_yolo12s_fp16.json`
- Modelos / Models:
  - `yolo12x.pt` (GT), `yolo12s.pt` (pred) — Ultralytics puede descargarlos automáticamente.
- OpenVINO export (si se genera):
  - `.../Images_to_evaluate/yolo12x_openvino_model/`
  - `.../Images_to_evaluate/yolo12s_fp16_openvino_model/`
- Logs TensorBoard:
  - `.../Images_to_evaluate/runs_all_in_one`
- Visualizaciones / Visualizations:
  - `.../Images_to_evaluate/eval_viz/<CAMERA>/`
- Plots:
  - `.../Images_to_evaluate/plots_metrics/`

Dependencias / Dependencies
- Python 3.8+
- pip install ultralytics torch tensorboard matplotlib
- (opcional) pip install opencv-python para visualizaciones / optional for visualizations
- (opcional) OpenVINO runtime si usa modelos exportados

Comandos básicos / Basic commands
- Ejecutar todo (GT + preds + evaluación):
  python3 /media/its/Stephany_120G/ACFR-Yolo-Pedestrian-Detection/eval_all_in_one.py
- Forzar regenerar GT y preds / Force regenerate:
  python3 .../eval_all_in_one.py --force-gt --force-pred
- Solo evaluar usando JSON existentes / Only evaluate:
  python3 .../eval_all_in_one.py --skip-pred-gen
- Guardar visualizaciones (ej. umbral pred específico) / Save visualizations:
  python3 .../eval_all_in_one.py --viz --viz-pred-conf 0.3
- Generar plots y barridos de umbrales / Generate plots:
  python3 .../eval_all_in_one.py --plot --gt-conf-list 0.25,0.5 --pred-conf-list 0.1,0.3,0.5
- Descargar pesos por URL (si no los tienes) / Download weights by URL:
  python3 .../eval_all_in_one.py --download-model --download-pred-model-url <URL_yolo12s.pt> --download-gt-model-url <URL_yolo12x.pt>
- Ver TensorBoard:
  tensorboard --logdir "/media/its/Stephany_120G/ACFR-Yolo-Pedestrian-Detection/Images_to_evaluate/runs_all_in_one"

Umbrales — qué aplican / Thresholds — what they apply to
- `pred_conf_list` (ej. 0.1, 0.2, 0.3): filtra cajas en el fichero de predicciones antes del matching — umbrales del test / filters predictions from pred JSON.
- `gt_conf` / `gt_conf_list`: filtra anotaciones dentro del pseudo-GT JSON (qué detecciones del pseudo-GT se consideran GT) antes del cálculo de métricas.
  - Recomendación: generar pseudo-GT con un umbral bajo (ej. 0.001) y luego evaluar con filtros GT más restrictivos si interesa.

Salida esperada — ejemplos / Expected output example
- Mensajes informativos al iniciar:
  [INFO] Imágenes en RGB: 4725
  [INFO] Imágenes en THERMAL: 3972
  [INFO] Total imágenes encontradas (todas las cámaras): 12689
  [INFO] Detectado pseudo-GT JSON existente: .../pseudo_gt_yolo12x.json. No se regenerará (usa --force-gt para forzar).
  [INFO] Detectado pred JSON existente: .../pred_yolo12s_fp16.json. No se regenerará (usa --force-pred para forzar).
- Resultados por umbral:
  [SUMMARY] conf_pred >= 0.1 -> TP=30333, FP=21872, FN=817, GT=31150, PRED=52205  
  [METRICS GLOBAL] precision=0.5932, recall=0.9737, f1=0.7341  
  [RESULT] AP@0.5 global = 0.9360

Interpretación rápida / Quick interpretation
- AP alto + precision baja → muchas detecciones correctas pero también bastantes FPs; ajustar umbral `pred_conf` para equilibrar.
- Usa TensorBoard para analizar métricas por umbral.

Notas útiles / Useful notes
- Si las carpetas de cámara están vacías, el script hace fallback a un scan recursivo en `Images_to_evaluate/` para encontrar imágenes.
- Visualizaciones:
  - Ruta por defecto: `.../Images_to_evaluate/eval_viz/<CAMERA>/`
  - Flag: `--viz` habilita guardado de visualizaciones TP/FP.
  - Límite por cámara: `--max-viz-per-cam` (eval_all_in_one.py) o `--max-viz` (infer_yolo.py). Por defecto 50. Ajusta si quieres menos/más.
  - Ejemplo: `python3 .../eval_all_in_one.py --viz --max-viz-per-cam 30` guardará hasta 30 visualizaciones por cámara.
  - Para verlas: abre las imágenes con tu visor de imágenes preferido o explora la carpeta.
- Plots:
  - Ruta por defecto de plots: `.../Images_to_evaluate/plots_metrics/` (o `--plots-outdir` para cambiar).
  - Flag: `--plot` genera gráficos (líneas y heatmaps) para las combinaciones de umbrales.
  - Ejemplo: `python3 .../eval_all_in_one.py --plot --plots-outdir /ruta/a/mis_plots`
- Otros:
  - Mantén la estructura de carpetas indicada y verifica permisos de escritura en `Images_to_evaluate/`.

Fin / End

