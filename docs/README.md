# Manuales de usuario — SAM3 LoRA

Manuales prácticos para usar este repositorio como un **método** que produce un LoRA especializado por clase sobre imágenes aéreas / satelitales.

## Índice

| # | Manual | Cuándo leerlo |
|---|---|---|
| 1 | [Manual de usuario](./MANUAL_USUARIO.md) | Primera lectura. Visión general, instalación, flujo de trabajo. |
| 2 | [Manual de preparación de dataset](./MANUAL_DATASET.md) | Antes de entrenar. Cómo dejar tu COCO listo. |
| 3 | [Manual de entrenamiento](./MANUAL_ENTRENAMIENTO.md) | Cuando entrenes un LoRA. Comando único + ajuste de hiperparámetros. |
| 4 | [Manual de inferencia](./MANUAL_INFERENCIA.md) | Después de entrenar. Predicción por tile + integración GIS. |
| 5 | [Manual de clases personalizadas](./MANUAL_CLASES.md) | Cuando agregues una clase nueva (incluyendo clases raras). |
| 6 | [Manual de troubleshooting](./MANUAL_TROUBLESHOOTING.md) | Cuando algo no funciona. |

## Glosario rápido

| Término | Significado |
|---|---|
| **SAM3** | *Segment Anything Model 3*. Modelo de Facebook que segmenta una imagen condicionado por un prompt de texto. |
| **LoRA** | *Low-Rank Adaptation*. Técnica que añade adaptadores pequeños (rank 16) al modelo base congelado. Sale un archivo de ~20–40 MB en lugar de re-entrenar todo. |
| **LoRA especialista** | Un archivo `.pt` entrenado para reconocer **una clase** (ej. solo construcciones, o solo árboles de aguacate). |
| **Prompt** | Texto que se le pasa a SAM3 para decirle qué buscar: `"building"`, `"avocado_tree"`, `"vegetation"`. |
| **Tile** | Recorte cuadrado de una imagen satelital grande (típicamente 1024×1024 px). |
| **COCO** | Formato JSON estándar para anotaciones (cajas + máscaras + categorías). |

## Mapa rápido del repositorio

```
SAM3_LoRA/
├── AERIAL_LORA_GUIDE.md         # Guía canónica en inglés (resumen ejecutivo)
├── docs/                         # Este manual de usuario en español
├── train_class_lora.py           # ★ Comando único de entrenamiento
├── predict_class.py              # ★ Comando único de inferencia
├── configs/aerial_class_lora.yaml  # Config parametrizado
├── prompts/class_prompts.yaml    # Registro de clases / sinónimos
├── scripts/check_install.py      # Pre-flight de dependencias
├── samples/tiny/                 # Slot para el smoke test
└── legacy/                       # Versiones anteriores (no se mantienen)
```
