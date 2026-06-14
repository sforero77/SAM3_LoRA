# Manual de inferencia

## 1. Comando único

```bash
python predict_class.py \
  --lora_weights outputs/avocado_lora/best_lora_weights.pt \
  --class_name avocado_tree \
  --input_dir data/test_tiles \
  --output_dir outputs/avocado_preds
```

Para cada tile en `--input_dir`, genera una máscara PNG binaria
(`<tile_stem>_mask.png`) bajo `<output_dir>/masks/` con la **unión de
todas las instancias detectadas** por encima del threshold (después de
NMS), a la **resolución original** del tile.

## 2. Argumentos

| Flag | Default | Notas |
|---|---|---|
| `--lora_weights` | (requerido) | Ruta al `.pt` entrenado. |
| `--class_name` | (requerido) | Texto del prompt. Idealmente igual a `--target_class` del entrenamiento. |
| `--input_dir` | (requerido) | Carpeta con tiles JPG/PNG/TIF. |
| `--output_dir` | (requerido) | Salida. Se crea si no existe. |
| `--config` | `<weights>/../resolved_config.yaml` | Override del config. |
| `--threshold` | 0.5 | Score mínimo para conservar una detección. |
| `--nms_iou` | 0.5 | Umbral de IoU para NMS. |
| `--coco_output` | False | Si lo pasas, también escribe `predictions.coco.json`. |

## 3. Qué hace predict_class.py paso a paso

1. **Carga el config** desde `resolved_config.yaml` (escrito por
   `train_class_lora.py` al lado de los pesos).
2. **Reconstruye el modelo**: SAM3 base + adaptadores LoRA con los
   mismos flags que se entrenó.
3. **Carga los pesos** del LoRA. El modelo base no se toca.
4. **Itera tile por tile**:
   - Resize a 1008×1008.
   - Forward con `query_text=class_name`.
   - Score threshold + NMS sobre las instancias predichas.
   - Re-escala las máscaras de 1008×1008 a la resolución original del
     tile (nearest neighbor).
   - Une todas las máscaras supervivientes en un único PNG binario.
5. **Guarda**:
   - `<output_dir>/masks/<tile_stem>_mask.png` — siempre.
   - `<output_dir>/predictions.coco.json` — solo si pasas `--coco_output`.

## 4. Ajuste de `--threshold` y `--nms_iou`

| Síntoma | Acción |
|---|---|
| Muy pocas detecciones | Bajar `--threshold` a 0.3 o 0.2. |
| Muchas detecciones falsas | Subir `--threshold` a 0.6–0.7. |
| Cajas duplicadas sobre el mismo objeto | Bajar `--nms_iou` a 0.3. |
| Objetos cercanos se fusionan | Subir `--nms_iou` a 0.7. |

Recomendación: empieza con defaults (0.5 / 0.5) y ajusta sobre una
muestra de validación.

## 5. Salida en COCO (--coco_output)

```json
{
  "images": [...],
  "categories": [{"id": 1, "name": "avocado_tree"}],
  "annotations": [
    {
      "id": 1, "image_id": 1, "category_id": 1,
      "bbox": [120, 340, 80, 60],
      "area": 4800,
      "score": 0.87,
      "iscrowd": 0,
      "segmentation": [[x1,y1,x2,y2,x3,y3,x4,y4]]
    }
  ]
}
```

⚠️ La `segmentation` del COCO de salida es un **rectángulo** envolvente
(fallback). La máscara real, pixel-precisa, está en el PNG bajo
`masks/`. Si necesitas polígonos exactos, usa OpenCV o
`skimage.measure.find_contours` sobre el PNG.

## 6. Integración con QGIS / GIS

El pipeline no maneja georreferenciación. Para reensamblar predicciones
en un raster geo-referenciado:

1. **Guarda metadatos por tile** cuando recortes:
   ```csv
   tile_name,         x_off,  y_off,  width,  height, crs,  affine
   tile_0001.png,     0,      0,      1024,   1024,   EPSG:32619,  ...
   tile_0002.png,     1024,   0,      1024,   1024,   EPSG:32619,  ...
   ```
2. **Después de inferir**, lee cada `<tile>_mask.png` y pégalo en el
   raster global usando `x_off`, `y_off`.
3. **Exporta como GeoTIFF** con rasterio:
   ```python
   import rasterio
   with rasterio.open("output_mask.tif", "w", **profile) as dst:
       dst.write(union_mask, 1)
   ```
4. **Para vectorizar** (de raster a shapefile / GeoJSON):
   ```python
   from rasterio.features import shapes
   for geom, val in shapes(mask, mask=mask>0, transform=affine):
       # geom es un dict GeoJSON-like
       ...
   ```

## 7. Inferencia con tiles solapados

Si recortaste tu raster con solape (recomendado para evitar artefactos
de borde), al reensamblar:

- **Voto mayoritario**: cada pixel del raster final pertenece a la clase
  si ≥ la mitad de los tiles que lo ven lo predijeron como positivo.
- **OR lógico** (más permisivo): pixel = positivo si **algún** tile lo
  predijo así.
- **AND lógico** (más conservador): pixel = positivo solo si **todos**
  los tiles que lo ven lo predijeron así.

OR lógico es el default razonable para vegetación / construcciones.

## 8. Procesar varios LoRAs en paralelo

Si tienes tres LoRAs (`building`, `vegetation`, `road`):

```bash
for cls in building vegetation road; do
  python predict_class.py \
    --lora_weights outputs/${cls}_lora/best_lora_weights.pt \
    --class_name $cls \
    --input_dir data/test_tiles \
    --output_dir outputs/${cls}_preds
done
```

Salida: tres carpetas de máscaras independientes. Para una capa
multi-clase, combínalas pixel a pixel con una regla de prioridad
(building > road > vegetation, por ejemplo).

## 9. Velocidad esperada

A100 40GB, tile 1024×1024:

| Operación | Tiempo |
|---|---|
| Carga del modelo (1 sola vez) | ~30 s |
| Forward por tile | 0.3–0.8 s |
| Post-proceso (NMS + resize máscaras) | ~0.1 s |
| **Throughput** | ~70–100 tiles / minuto |

Para inferencia a gran escala (10k+ tiles), considera:
- Procesar en batch (modificar `predict_class.py` para `--batch 4`).
- Pre-cargar todos los tiles en RAM antes de iterar.
- TensorRT / `torch.compile` (no probado aquí, pero compatible).

## 10. Baseline: SAM3 base (sin LoRA)

Para tener un punto de referencia (qué tan bien lo hace SAM3 sin afinar),
evalúa el modelo base sobre tu set de validación:

```bash
python validate_sam3_lora.py \
  --val_data_dir data/avocado/valid \
  --use-base-model
```

No requiere `--config` ni `--weights`. Reporta cgF1 / mAP del SAM3
original. Si tu LoRA no mejora estas métricas, algo está mal en el
entrenamiento. Para una comparación **visual** lado-a-lado base vs. LoRA,
usa `compare_lora_base.py` (sección 11).

## 11. Visualización rápida (debug)

`inference_lora.py` (el módulo subyacente) tiene un `visualize_predictions()`
que dibuja bboxes y overlays de máscaras sobre el tile original.
Para usarlo:

```bash
python inference_lora.py \
  --config outputs/avocado_lora/resolved_config.yaml \
  --weights outputs/avocado_lora/best_lora_weights.pt \
  --image data/test_tiles/tile_0001.png \
  --prompt "avocado_tree" \
  --output debug_vis.png
```

Genera un PNG con bboxes en rojo + overlay rojo translúcido de las
máscaras. Útil para revisar resultados sin escribir código.
