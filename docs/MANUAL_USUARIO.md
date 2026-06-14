# Manual de usuario

## 1. Qué hace este repositorio

Te permite entrenar un **LoRA especializado por clase** a partir de SAM3
(Segment Anything Model 3), usando tu propio dataset de imágenes aéreas /
satelitales en formato COCO.

> **Un LoRA = una clase.** Si quieres tres clases (construcciones,
> vegetación, vías), entrenas tres LoRAs separados. Al desplegar eliges
> cuál usar según el caso.

Cada LoRA pesa ~20–40 MB (vs. ~2 GB del modelo base), reusa el modelo
SAM3 cargado en memoria, y al inferir le pasas el texto de la clase
como prompt (`"building"`, `"avocado_tree"`, etc.).

## 2. Requisitos

| Componente | Mínimo recomendado |
|---|---|
| GPU NVIDIA | ≥ 16 GB VRAM (A100, A6000, RTX 6000 Ada). Para batch >1 conviene 24+ GB. |
| RAM | 32 GB |
| Disco | 50 GB (modelo base + checkpoints + datasets) |
| Python | 3.10 o 3.11 |
| PyTorch | ≥ 2.7 |
| Acceso internet | Primera descarga del modelo `facebook/sam3` desde Hugging Face. |

> **Sistema operativo:** el entrenamiento requiere **Linux + CUDA**. La ruta
> de pérdida usa kernels de Triton, que no tienen wheels para Windows; en
> Windows el código sigue siendo importable (fallbacks en PyTorch puro) pero
> el entrenamiento igual necesita una GPU CUDA. No hay ruta de CPU usable.

## 3. Instalación

> **Antes de nada — acceso a SAM3 (obligatorio).** `facebook/sam3` es un
> modelo *gated* en Hugging Face. Entra a
> <https://huggingface.co/facebook/sam3>, pulsa **Request Access** y acepta
> la licencia. La aprobación suele ser rápida pero es **obligatoria**: sin
> ella el primer entrenamiento/inferencia falla con `GatedRepoError`
> (HTTP 401/403) aunque hayas hecho login.

```bash
git clone https://github.com/sforero77/SAM3_LoRA.git
cd SAM3_LoRA

# (opcional) entorno aislado
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Autentícate en Hugging Face (después de tener el acceso aprobado)
hf auth login        # o:  export HF_TOKEN=tu_token

# Verifica que todo importa y CUDA esté visible
python scripts/check_install.py
```

`scripts/check_install.py` no carga el modelo (es pesado), pero
confirma que las dependencias, el config template, el registro de
prompts y CUDA estén disponibles. Si alguna línea da ❌, arreglas eso
antes de continuar.

## 4. Flujo de trabajo end-to-end

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────┐
│ 1. Preparar │───▶│ 2. Entrenar      │───▶│ 3. Inferir   │
│   dataset   │    │   LoRA por clase │    │   por tile   │
│   (COCO)    │    │   (~horas)       │    │   (segundos) │
└─────────────┘    └──────────────────┘    └──────────────┘
```

### 4.1 Preparar dataset

Detalle completo en **[MANUAL_DATASET.md](./MANUAL_DATASET.md)**. Resumen:

```
<dataset_root>/
  train/
    images/                       # tiles RGB 1024×1024
    _annotations.coco.json
  valid/
    images/
    _annotations.coco.json
```

Requisitos clave del COCO:
- `categories[].name` debe coincidir con `--target_class`.
- Cada `annotations[]` debe traer `bbox`, `category_id` y `segmentation`
  (polígono o RLE — sin esto el LoRA solo aprende cajas).

### 4.2 Entrenar

Detalle en **[MANUAL_ENTRENAMIENTO.md](./MANUAL_ENTRENAMIENTO.md)**.
Comando único:

```bash
python train_class_lora.py \
  --dataset_root data/avocado \
  --target_class avocado_tree \
  --output_dir outputs/avocado_lora
```

Salida: `outputs/avocado_lora/best_lora_weights.pt` (mejor val loss),
`last_lora_weights.pt` (último epoch), `resolved_config.yaml` (config
resuelto — lo usa inferencia).

### 4.3 Inferir

Detalle en **[MANUAL_INFERENCIA.md](./MANUAL_INFERENCIA.md)**.
Comando único:

```bash
python predict_class.py \
  --lora_weights outputs/avocado_lora/best_lora_weights.pt \
  --class_name avocado_tree \
  --input_dir data/test_tiles \
  --output_dir outputs/avocado_preds
```

Salida: un PNG binario por tile bajo `outputs/avocado_preds/masks/` con
la unión de todas las detecciones por encima del umbral.

## 5. Añadir una clase nueva

Tres pasos, sin tocar código:

1. **Registrar** la clase en `prompts/class_prompts.yaml`:
   ```yaml
   greenhouse:
     synonyms: [greenhouse, invernadero, plastic greenhouse]
     description: [transparent rectangular structure, agricultural plastic cover]
   ```

2. **Preparar** tu COCO con `categories[].name = "greenhouse"`.

3. **Entrenar** con `--target_class greenhouse`.

Detalle en **[MANUAL_CLASES.md](./MANUAL_CLASES.md)**.

## 6. Smoke test antes de gastar GPU-hours

Ver sección 6 del [AERIAL_LORA_GUIDE.md](../AERIAL_LORA_GUIDE.md) o
`samples/tiny/README.md`. Resumen: genera el dataset de prueba con
`python scripts/make_tiny_sample.py` (5 + 2 tiles sintéticos con máscaras,
sin etiquetar nada a mano), entrena 3 epochs, y en menos de 5 minutos debe
producir un LoRA < 50 MB y una predicción no vacía. Si esto no pasa, el
problema está en tu setup, no en el modelo.

## 7. Si algo no funciona

Ver **[MANUAL_TROUBLESHOOTING.md](./MANUAL_TROUBLESHOOTING.md)**.
