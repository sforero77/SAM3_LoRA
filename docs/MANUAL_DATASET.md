# Manual de preparación de dataset

## 1. Estructura obligatoria

```
<dataset_root>/
  train/
    images/                       # *.jpg | *.png | *.tif (RGB)
    _annotations.coco.json
  valid/
    images/
    _annotations.coco.json
```

El loader lee `<dataset_root>/<split>/_annotations.coco.json` y
`<dataset_root>/<split>/images/`. Estos nombres son fijos.

## 2. Tiles (las imágenes)

- **Formato**: JPG o PNG. RGB de 3 canales. Sin canal alfa, sin bandas
  extra (NIR, SWIR, etc. — elimínalas antes).
- **Tamaño recomendado**: 1024×1024 px. El pipeline las redimensiona
  internamente a 1008×1008 (resolución nativa de SAM3). Cualquier
  tamaño cuadrado funciona, pero tiles muy grandes (4k+) requieren
  recorte previo.
- **Profundidad de bit**: 8-bit. Si tu imagen viene en 16-bit (típico de
  Sentinel/Maxar), normaliza a 0–255 antes.
- **Sin georreferenciación dentro del tile.** Conserva los metadatos
  geo aparte (CSV / sidecars) para reensamblar después de inferir.
- **Nombrado consistente**: el `file_name` del COCO debe coincidir
  exactamente con el archivo en `images/`. Sin paths relativos en el
  COCO; solo el basename.

### Cómo recortar tu raster grande en tiles

Esto se hace fuera del pipeline. Herramientas típicas:

- **`gdal_retile.py`** (GDAL): recorta un GeoTIFF en tiles de tamaño fijo.
- **`rasterio.windows`**: para scripts Python custom.
- **QGIS** *"Raster → Extraction → Clip Raster by Extent"*: GUI.

Recomendación: tiles solapados 10–20% si necesitas predicciones limpias
al reensamblar (los bordes del tile suelen tener peor calidad).

## 3. Anotaciones (COCO)

### 3.1 Estructura del JSON

```json
{
  "images": [
    {"id": 1, "file_name": "tile_0001.png", "width": 1024, "height": 1024}
  ],
  "categories": [
    {"id": 1, "name": "building"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [120, 340, 80, 60],
      "area": 4800,
      "iscrowd": 0,
      "segmentation": [[120,340, 200,340, 200,400, 120,400]]
    }
  ]
}
```

### 3.2 Campos obligatorios

| Sección | Campo | Tipo | Notas |
|---|---|---|---|
| `images[]` | `id` | int | Único. |
| | `file_name` | str | Basename solo. Debe existir en `images/`. |
| | `width`, `height` | int | Dimensiones reales del archivo. |
| `categories[]` | `id` | int | Único. |
| | `name` | str | **Importantísimo**: este string se usa como prompt. |
| `annotations[]` | `image_id` | int | Apunta a `images[].id`. |
| | `category_id` | int | Apunta a `categories[].id`. |
| | `bbox` | `[x,y,w,h]` | Pixeles de la imagen original. |
| | `segmentation` | polígono o RLE | **Obligatorio**. Ver sección 3.3. |

Opcionales: `area` (recomendado), `iscrowd` (default 0), `id` por anotación.

### 3.3 El campo `segmentation`

Acepta dos formatos. Ambos los maneja `pycocotools` automáticamente.

**Polígono** (lista de listas de coordenadas):
```json
"segmentation": [[x1, y1, x2, y2, x3, y3, ...]]
```
Cada sublista es un polígono. Múltiples sublistas = objeto con hueco
o polígono multi-pieza. Coordenadas en pixeles de la imagen original.

**RLE COCO** (run-length encoded, eficiente para máscaras complejas):
```json
"segmentation": {
  "counts": [10, 5, 20, 5, ...],
  "size": [height, width]
}
```
O en formato comprimido (string base64):
```json
"segmentation": {"counts": "X1X2...", "size": [1024, 1024]}
```

**Sin `segmentation` válida el LoRA solo aprende cajas**, lo cual es
inútil para clases donde el borde importa (vegetación, construcciones,
agua). El pipeline detecta esto, loguea un warning, y el collator
desactiva la pérdida de máscara para esa anotación específica.

## 4. Herramientas para anotar

### 4.1 Roboflow (recomendado para empezar)

- Sube tus tiles → anota con polígonos → exporta como `COCO Segmentation`.
- Roboflow ya genera la estructura `train/`, `valid/`, `test/` con
  `_annotations.coco.json` lista para usar.
- Pega `train/` y `valid/` directo bajo `<dataset_root>/`.

### 4.2 CVAT (open source, autoalojado)

- Útil cuando tienes mucho volumen y necesitas equipo.
- Exporta como **COCO 1.0**.
- Revisar que el JSON tenga `segmentation` con polígonos, no solo
  `bbox` (depende del modo de anotación elegido).

### 4.3 QGIS + plugin AI

- Si tu fuente es un shapefile o GeoJSON, conviene rasterizar polígonos
  por tile y convertirlos a COCO.
- Script base: para cada tile, intersecta los polígonos con el extent
  del tile, convierte coordenadas geográficas → pixel coords del tile,
  y guarda como anotación COCO.

### 4.4 LabelMe / VIA / Supervisely

- Cualquier herramienta que exporte polígonos es válida. Si no exporta
  COCO directamente, hay scripts puente (`labelme2coco`, `via2coco`).

## 5. Convertir desde otros formatos

El repo ya incluye:

- `convert_roboflow_to_coco.py` — Roboflow export → COCO consolidado.
- `prepare_data.py` — utilidades para YOLO ↔ COCO, validación de
  dataset, conversión de máscaras.

Si tienes shapefiles georreferenciados, tu pipeline custom debe:

1. Iterar tile a tile.
2. Para cada tile, encontrar polígonos que intersecten su extent.
3. Reproyectar al CRS del raster, luego convertir a pixel coords del
   tile (usando el affine transform).
4. Escribir el polígono como `[[x1,y1,x2,y2,...]]` en el COCO.

## 6. Validación rápida del dataset

```bash
python prepare_data.py validate \
  --data_dir <dataset_root> \
  --split train
```

Verifica que cada imagen tenga su anotación, el JSON sea válido y los
bboxes tengan 4 valores. No verifica la calidad de la segmentación;
para eso, abre algunas anotaciones en Roboflow / QGIS.

## 7. Sanidad mínima

| Métrica | Recomendación |
|---|---|
| Tiles de train con la clase objetivo | ≥ 200 |
| Tiles de valid con la clase objetivo | ≥ 50 |
| Tiles vacíos (sin instancias) en cada split | < 30 % |
| Instancias por tile (mediana) | 1–20 (sobre 50 ralentiza el batch) |
| Tamaño promedio de instancia | ≥ 20×20 px (objetos < 10 px son difíciles) |

Si tu clase es muy esparcida (ej. 1 instancia cada 10 tiles), considera
recortar tiles más pequeños o pre-filtrar tiles vacíos.

## 8. Multi-clase con LoRAs separados

Un solo COCO puede contener varias clases:

```json
"categories": [
  {"id": 1, "name": "building"},
  {"id": 2, "name": "vegetation"},
  {"id": 3, "name": "road"}
]
```

Para entrenar tres LoRAs distintos:

```bash
python train_class_lora.py --dataset_root data/ortho --target_class building   --output_dir outputs/building_lora
python train_class_lora.py --dataset_root data/ortho --target_class vegetation --output_dir outputs/vegetation_lora
python train_class_lora.py --dataset_root data/ortho --target_class road       --output_dir outputs/road_lora
```

El filtro `--target_class` conserva solo esa categoría y descarta tiles
sin instancias después del filtro. No reutiliza tiles entre LoRAs;
cada entrenamiento ve solo lo de su clase.

## 9. Ejemplo de COCO mínimo viable

Útil para el smoke test de `samples/tiny/`. Reemplaza polígonos por los
reales y file_names por los tuyos:

```json
{
  "images": [
    {"id": 1, "file_name": "tile_0001.png", "width": 1024, "height": 1024},
    {"id": 2, "file_name": "tile_0002.png", "width": 1024, "height": 1024}
  ],
  "categories": [
    {"id": 1, "name": "building"}
  ],
  "annotations": [
    {
      "id": 1, "image_id": 1, "category_id": 1,
      "bbox": [100, 100, 120, 80], "area": 9600, "iscrowd": 0,
      "segmentation": [[100,100, 220,100, 220,180, 100,180]]
    },
    {
      "id": 2, "image_id": 1, "category_id": 1,
      "bbox": [300, 400, 150, 100], "area": 15000, "iscrowd": 0,
      "segmentation": [[300,400, 450,400, 450,500, 300,500]]
    },
    {
      "id": 3, "image_id": 2, "category_id": 1,
      "bbox": [50, 50, 200, 200], "area": 40000, "iscrowd": 0,
      "segmentation": [[50,50, 250,50, 250,250, 50,250]]
    }
  ]
}
```
