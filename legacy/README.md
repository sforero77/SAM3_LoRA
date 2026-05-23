# Legacy

Esta carpeta contiene scripts, configs y documentación de iteraciones anteriores
del repo que ya no forman parte del pipeline canónico.

Se conservan como referencia, pero **no se mantienen** y pueden quedar obsoletos.

El pipeline activo está documentado en `AERIAL_LORA_GUIDE.md` (raíz del repo).

## Contenido

- `scripts/` — variantes anteriores de scripts de entrenamiento (train.py, train_native.py,
  train_sam3_lora.py, train_sam3_lora_native.py, train_simple.py, train_standalone.py).
- `configs/` — configs YAML antiguos (crack_detection, light/minimal/full lora, base,
  standalone) y carpeta `sam3_lora_configs/` con configs Hydra nativos de SAM3.
- `docs/` — guías y resúmenes anteriores (CLI_TRAINING_GUIDE, LORA_IMPLEMENTATION_GUIDE,
  QUICK_SUMMARY, PROJECT_SUMMARY, README_LORA_IMPLEMENTATION, README_INFERENCE,
  diagnose_training, PROJECT_STRUCTURE).

## Pipeline canónico (activo)

- `train_sam3_lora_with_categories.py` — trainer subyacente.
- `train_class_lora.py` — wrapper CLI canónico (a añadir en Fase 2).
- `predict_class.py` — inferencia por clase (a añadir en Fase 4).
- `configs/aerial_class_lora.yaml` — config único (a añadir en Fase 2).
- `prompts/class_prompts.yaml` — registro de prompts por clase.
