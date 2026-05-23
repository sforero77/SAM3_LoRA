# Manual de entrenamiento

## 1. Comando único

```bash
python train_class_lora.py \
  --dataset_root data/avocado \
  --target_class avocado_tree \
  --output_dir outputs/avocado_lora
```

Lo que pasa internamente:

1. **Validación del contrato** — verifica `train/` y `valid/`, su
   `_annotations.coco.json`, y que `categories[].name` incluya la clase
   objetivo. Falla rápido con error claro si algo no cuadra.
2. **Resolución del config** — toma `configs/aerial_class_lora.yaml`,
   sustituye `{{TARGET_CLASS}}` y `{{DATASET_ROOT}}`, lo guarda en
   `<output_dir>/resolved_config.yaml` para que la inferencia lo reuse.
3. **Carga del dataset** — filtra a la clase objetivo (descarta otras
   categorías y tiles sin instancias remanentes). Aplica augmentación
   en `train/`, deterministico en `valid/`.
4. **Carga del modelo** — descarga `facebook/sam3` de Hugging Face la
   primera vez (~2 GB), aplica adaptadores LoRA con los flags del config.
5. **Entrenamiento** — Adam + cosine schedule, mixed precision bf16.
6. **Guardado** — `best_lora_weights.pt` (mejor val loss) y
   `last_lora_weights.pt` cada epoch.

## 2. Argumentos del CLI

| Flag | Default | Notas |
|---|---|---|
| `--dataset_root` | (requerido) | Carpeta con `train/` y `valid/`. |
| `--target_class` | (requerido) | Debe coincidir (case-insensitive) con `categories[].name`. |
| `--output_dir` | (requerido) | Dónde van pesos, logs y config resuelto. |
| `--config` | `configs/aerial_class_lora.yaml` | Override del template. |
| `--epochs` | 40 (del config) | Override rápido. |
| `--batch_size` | 1 (del config) | Más alto requiere más VRAM. |
| `--learning_rate` | 1e-4 (del config) | Override. |
| `--skip_validation` | False | No recomendado. Saltea el chequeo de contrato. |

## 3. Defaults del config

`configs/aerial_class_lora.yaml`:

```yaml
lora:
  rank: 16              # Capacidad del adaptador. 8 = más liviano, 32 = más capacidad
  alpha: 32             # Típicamente 2 × rank
  dropout: 0.05         # Regularización modesta
  apply_to_vision_encoder: true
  apply_to_text_encoder: true        # CRÍTICO para clases raras
  apply_to_geometry_encoder: true
  apply_to_detr_encoder: true
  apply_to_detr_decoder: true
  apply_to_mask_decoder: true

training:
  batch_size: 1
  gradient_accumulation_steps: 4     # Batch efectivo = 4
  learning_rate: 1.0e-4
  weight_decay: 0.01
  num_epochs: 40
  warmup_steps: 100
  lr_scheduler: cosine
  mixed_precision: bf16
```

## 4. Cuándo cambiar qué hiperparámetro

### Si vas corto de VRAM (OOM)

| Cambio | Efecto |
|---|---|
| `batch_size: 1` (ya está) | Mínimo posible. |
| `gradient_accumulation_steps: 8` | Batch efectivo 8, mismo VRAM. |
| `lora.rank: 8`, `alpha: 16` | Menos parámetros entrenables. |
| `apply_to_geometry_encoder: false` | Ahorra algo. |
| Reducir resolución de tiles a 768×768 | (Requiere cambiar `self.resolution` en el dataset). |

### Si el modelo no aprende (val loss plana)

| Cambio | Cuándo |
|---|---|
| Verificar que `segmentation` no esté vacío en tu COCO | Lo más común. |
| Subir `learning_rate` a `3e-4` | Pocas instancias por epoch. |
| Subir `num_epochs` a 80–100 | Dataset pequeño (< 500 tiles). |
| Activar `apply_to_text_encoder: true` (ya está) | Clase rara cuyo embedding inicial es débil. |
| Bajar `dropout` a 0.0 | Dataset muy pequeño que está bloqueando aprendizaje. |

### Si el modelo sobreajusta (train loss bajando, val subiendo)

| Cambio | Cuándo |
|---|---|
| Más augmentación (ya hay 6 ops geométricas + jitter) | Default ya razonable. |
| Subir `dropout` a 0.1–0.2 | Sobreajuste claro. |
| Bajar `lora.rank` a 8 | Adaptador con menos capacidad. |
| `weight_decay: 0.05` | Regularización L2 más fuerte. |
| Más datos en `train/` | Mejor solución de fondo. |

### Si la val loss oscila

| Cambio | Cuándo |
|---|---|
| `gradient_accumulation_steps: 8` o 16 | Batch efectivo mayor reduce varianza. |
| `learning_rate` a la mitad | Pasos muy grandes. |

## 5. Augmentación geométrica

Aplicada solo en `train/`, **una misma op por sample** consistente
sobre imagen, bbox y máscara. Las 6 ops son:

| Op | Geometría |
|---|---|
| `identity` | sin cambios |
| `hflip` | espejo horizontal |
| `vflip` | espejo vertical |
| `rot90` | 90° horario |
| `rot180` | 180° |
| `rot270` | 90° antihorario |

Esto multiplica efectivamente tu dataset ×6 sin coste adicional de
labeling. Es seguro porque las imágenes aéreas son nadir (no hay
horizonte que sesgar). Para datasets street-view o frontales esto
**no** sería apropiado.

Jitter fotométrico (brillo / contraste ±15%) también en `train/` solo.
Modesto a propósito: la reflectancia es informativa en imágenes
satelitales.

## 6. Augmentación de prompts

En cada paso del epoch, el prompt enviado a SAM3 se muestrea
aleatoriamente del registro `prompts/class_prompts.yaml`:

```yaml
avocado_tree:
  synonyms: [avocado tree, árbol de aguacate, aguacate]
  description: [dense round canopy, dark green tree]
```

El LoRA aprende el **concepto visual** asociado a varios términos.
Beneficios:

- Robusto a paráfrasis al desplegar.
- Soporta queries multilingües sin reentrenar.
- Para clases raras, los `description` añaden señal visual que ancla
  el embedding del texto.

En `valid/` siempre se usa el primer sinónimo (canónico). Esto hace
el val loss reproducible entre runs.

## 7. Salida del entrenamiento

```
outputs/avocado_lora/
├── best_lora_weights.pt          # ★ Mejor val loss — usa este para inferir
├── last_lora_weights.pt          # Último epoch
├── resolved_config.yaml          # ★ Config resuelto, lo necesita predict_class.py
├── logs/                          # Logs de training
└── ...
```

Tamaño típico del `.pt`: 20–40 MB (LoRA rank 16, todos los
componentes). Si pesa mucho más, revisa que `save_lora_only: true` en
el config (default).

## 8. Tiempos esperados

Aproximaciones para una **A100 40GB**:

| Tiles train | Epochs | Tiempo total |
|---|---|---|
| 200 | 40 | ~30 min |
| 800 | 40 | ~2 h |
| 2000 | 40 | ~5 h |
| 5000 | 40 | ~12 h |

Multiplica ×2–3 para RTX 4090 / A6000, ×5+ para CPU (no recomendado).

## 9. Monitoreo

Stdout muestra:
- Loss por step (`pbar.set_postfix`)
- Loss promedio por epoch (train y val)
- Cuándo se guarda un nuevo mejor checkpoint

Para tensorboard / W&B: integrarlos sería un cambio menor en
`train_sam3_lora_with_categories.py` (línea ~493 del `train_epoch`).
Por defecto el repo no los activa.

## 10. Validación post-entrenamiento

Para métricas formales (cgF1, mAP, mIoU):

```bash
python validate_sam3_lora.py \
  --config outputs/avocado_lora/resolved_config.yaml \
  --weights outputs/avocado_lora/best_lora_weights.pt \
  --val_data data/avocado/valid
```

Este script no es parte del flujo canónico — es un benchmark que se
ejecuta una vez al final. Reporta cgF1 (categoría-aware F1, métrica
nativa de SAM3), cgF1@50, cgF1@75 y mAP.

## 11. Comparar LoRA vs. base

```bash
python compare_lora_base.py \
  --config outputs/avocado_lora/resolved_config.yaml \
  --weights outputs/avocado_lora/best_lora_weights.pt \
  --image samples/tiny/valid/images/tile_0001.png \
  --prompt "avocado_tree"
```

Genera lado-a-lado del SAM3 base vs. SAM3+LoRA. Útil para verificar
visualmente que el adaptador esté aportando algo.
