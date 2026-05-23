# Manual de troubleshooting

## 1. Errores en la preparación / validación

### `target_class='X' not found in COCO categories. Available: [...]`
- **Causa**: el `--target_class` que pasaste no coincide (case-insensitive)
  con ningún `categories[].name` del COCO.
- **Fix**: revisa la lista que imprime el error y usa exactamente uno
  de esos nombres, o renombra la categoría en tu COCO.

### `No images left in <split>/images after filtering`
- **Causa**: filtraste por una clase que existe en `categories` pero
  no tiene anotaciones en ese split.
- **Fix**: confirma que hay anotaciones con ese `category_id` en
  `train/` y `valid/`. Si no, etiqueta más tiles o usa otro split.

### `Missing COCO file: <path>/_annotations.coco.json`
- **Causa**: estructura de carpetas incorrecta.
- **Fix**: verifica que sea exactamente `<root>/train/_annotations.coco.json`
  y `<root>/valid/_annotations.coco.json`. Los nombres son fijos.

### `Invalid JSON in: _annotations.coco.json`
- **Causa**: archivo corrupto o mal formado.
- **Fix**: `python -c "import json; json.load(open('path/to/_annotations.coco.json'))"`
  para ver el error exacto. Reanota o regenera el archivo.

### Warning: `Segmentation decode failed for X ann#Y`
- **Causa**: pycocotools no pudo parsear ese `segmentation`. Polígono
  con menos de 3 puntos, RLE corrupto, coordenadas fuera de la imagen.
- **Fix**: la anotación afectada se entrena solo con bbox; las demás
  siguen normales. Si pasa en muchas anotaciones, revisa el script que
  generó el COCO.

### Predicciones son solo cajas, no bordes
- **Causa**: tu COCO no tiene `segmentation` válida en la mayoría de
  anotaciones. El LoRA solo aprendió bboxes.
- **Fix**: re-anotar con polígonos. Sin máscaras, el modelo no puede
  aprender bordes precisos.

## 2. Errores en el entrenamiento

### `CUDA out of memory`
- **Causa**: VRAM insuficiente para el batch + rank de LoRA + tiles 1008×1008.
- **Fix por orden de impacto**:
  1. `--batch_size 1` (debe ser el default).
  2. Subir `gradient_accumulation_steps` a 8 o 16 en el config.
  3. Bajar `lora.rank` a 8 y `alpha` a 16.
  4. Desactivar `apply_to_geometry_encoder: false` en el config.
  5. Bajar resolución (requiere editar `self.resolution = 1008` en el dataset).

### Loss = NaN después de unos pasos
- **Causa**: gradient explosion, lr demasiado alto, o bf16 inestable.
- **Fix**:
  1. Bajar `learning_rate` a 5e-5.
  2. Subir `warmup_steps` a 500.
  3. Cambiar `mixed_precision: bf16` a `fp32` (gasta más VRAM pero estable).

### Val loss baja muy lentamente / no baja
- **Causa**: dataset muy pequeño, lr muy bajo, o problema en `segmentation`.
- **Fix**:
  1. Confirma que el LoRA está cargado: el log debe imprimir
     "Trainable parameters: ~106K (X%)".
  2. Verifica visualmente algunas anotaciones — abre el COCO en
     Roboflow / CVAT.
  3. Sube lr a 3e-4 si dataset < 500 tiles.
  4. Aumenta epochs a 80.

### Train loss baja pero val loss sube (sobreajuste)
- **Fix**:
  1. Subir `dropout` a 0.1 o 0.15.
  2. Bajar `lora.rank` a 8.
  3. Más datos en train.
  4. Confirmar que augmentación esté activa (debe imprimir
     "🔄 Augmentation enabled" al cargar el train_dataset).

### "Model already has LoRA applied" / no se aplica LoRA
- **Causa**: re-aplicaste LoRA sobre un modelo ya adaptado.
- **Fix**: reinicia el proceso. No reuses el objeto `model` entre runs
  sin reconstruirlo.

### Modelo de Hugging Face no descarga
- **Causa**: sin internet, sin token de HF, o repo privado.
- **Fix**:
  1. `huggingface-cli login` para autenticarte.
  2. `export HF_HUB_OFFLINE=0`.
  3. Si estás detrás de proxy, exportar `HTTP_PROXY` y `HTTPS_PROXY`.

## 3. Errores en la inferencia

### `No --config given and resolved_config.yaml not found`
- **Causa**: el LoRA fue entrenado fuera de `train_class_lora.py` (no
  hay `resolved_config.yaml` al lado de los pesos).
- **Fix**: pasa `--config <path/al/yaml/que/usaste>` manualmente, o
  re-entrena con el wrapper canónico.

### Inferencia ultra-lenta (segundos/tile)
- **Causa esperada**: 0.3–0.8 s por tile en A100. Si tarda más:
- **Fix**:
  1. Verifica que `torch.cuda.is_available()` sea True
     (`python scripts/check_install.py`).
  2. Confirma que no estás corriendo en CPU.
  3. Para volumen alto (10k+ tiles), modifica `predict_class.py` para
     hacer batching de 4–8 tiles por forward.

### Predicciones vacías (todos los PNG son negros)
- **Causa**: threshold muy alto, o el modelo no detecta nada.
- **Fix**:
  1. Bajar `--threshold` a 0.2 o 0.1 y revisar.
  2. Si sigue vacío, comparar con SAM3 base:
     ```
     python infer_sam.py --image tile_0001.png --prompt "building"
     ```
     Si SAM3 base detecta, tu LoRA está mal. Si tampoco detecta, el
     dataset/prompt no funciona.
  3. Revisar visualmente con `inference_lora.py` (sección 11 del
     MANUAL_INFERENCIA).

### Predicciones con muchos falsos positivos
- **Fix**:
  1. Subir `--threshold` a 0.6–0.7.
  2. Bajar `--nms_iou` a 0.3.
  3. Si persiste, revisar dataset de entrenamiento — falsos positivos
     suelen indicar que el LoRA aprendió a marcar TODO como la clase.
     Quizá tu dataset tiene poco contraste (todos los tiles tienen la
     clase, sin tiles negativos).

### `Shape mismatch when loading LoRA weights`
- **Causa**: el config usado al inferir no coincide con el del
  entrenamiento (rank distinto, target_modules distintos, o flags
  apply_to_* distintos).
- **Fix**: usa `resolved_config.yaml` que se guarda automáticamente al
  lado de los pesos.

## 4. Problemas con el registro de prompts

### Warning: `Class 'X' not in prompt registry`
- **Causa**: tu clase no está en `prompts/class_prompts.yaml`.
- **Fix**: no es fatal — usa el nombre de COCO tal cual. Para mejor
  robustez, añade entrada al YAML (sección 3 del MANUAL_CLASES).

### Sinónimos no se aplican durante el entrenamiento
- **Verifica**:
  1. El nombre de tu clase en el YAML coincide (lowercase) con
     `categories[].name`.
  2. El log al inicio del train dice "🔄 Augmentation enabled".
  3. En modo eval (val_dataset) sí se usa siempre el canónico — es por
     diseño, no es un bug.

## 5. Problemas de entorno / instalación

### `ModuleNotFoundError: No module named 'pycocotools'`
- **Fix**: `pip install pycocotools` (ya está en `requirements.txt`).

### `ModuleNotFoundError: No module named 'sam3'`
- **Causa**: corres desde otra carpeta.
- **Fix**: ejecuta los scripts **desde la raíz del repo**, no desde
  `legacy/` o `scripts/`.

### `torch.cuda.is_available() == False`
- **Causa**: torch instalado sin soporte CUDA, o drivers desactualizados.
- **Fix**: reinstalar torch con la build correcta:
  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

### `decord` falla al instalar
- **Causa**: `decord` requiere libs de sistema en algunos entornos.
- **Fix**: este pipeline no usa video, así que puedes comentar
  `decord>=0.6.0` en `requirements.txt`.

## 6. Cómo reportar un bug

Cuando algo no funciona y no encuentras el síntoma aquí, junta:

1. **Comando exacto** que corriste.
2. **Stdout completo** desde el primer error hacia atrás (50 líneas).
3. **`resolved_config.yaml`** del run.
4. **Versión**: `git rev-parse HEAD` + `pip freeze | grep -E "torch|transformers|pycocotools"`.
5. **Dataset stats**: cuántos tiles en train/valid, cuántas
   anotaciones, cuántas tienen `segmentation`.
6. **Smoke test**: ¿funcionó la receta de `samples/tiny/`? Si no,
   incluye su error también.

Con eso se puede diagnosticar rápido. Sin eso, hay que adivinar.
