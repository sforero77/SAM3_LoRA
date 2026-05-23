# Manual de clases personalizadas

## 1. Cómo funciona la "clase" en este pipeline

SAM3 es **open-vocabulary**: no tiene una lista fija de clases. Lo que
hace es:

1. Codificar tu prompt de texto (`"avocado_tree"`, `"invernadero"`,
   lo que sea) con un text encoder tipo CLIP.
2. Cruzar ese embedding con la imagen.
3. Producir máscaras de las regiones que coinciden con el concepto.

Tu LoRA **adapta** ese cruce para una clase específica usando tus
ejemplos etiquetados. Por eso un LoRA = una clase.

> **El nombre de tu clase ES el prompt.** Si en el COCO la categoría
> se llama `"avocado_tree"`, al inferir tienes que pasar
> `--class_name avocado_tree`. Mismatch = pérdida de precisión.

## 2. Granularidad: tú decides

Una sola dimensión, tú la fijas:

| Granularidad | Ejemplo |
|---|---|
| Genérica | `vegetation` |
| Sub-clase | `tree` |
| Específica | `avocado_tree`, `lemon_tree` |
| Muy específica | `young_avocado_tree`, `mature_avocado_tree` |

Cada nivel requiere su propio LoRA y su propio COCO. Más específico =
menos datos disponibles típicamente = más difícil. Empieza por la
granularidad más gruesa que te resuelva el problema.

## 3. El registro `prompts/class_prompts.yaml`

Es donde declaras las clases que entiende el pipeline. Estructura:

```yaml
<nombre_canónico>:
  parent: <opcional, solo documentación>
  synonyms: [<lista de sinónimos>]
  description: [<opcional, descriptores visuales>]
```

### 3.1 Campos

- **Clave (nombre_canónico)**: debe coincidir con `categories[].name`
  del COCO. Case-insensitive.
- **`synonyms`**: lista. El **primero** es el canónico (se usa en
  evaluación / inferencia por defecto). Los demás se muestrean
  aleatoriamente durante el entrenamiento.
- **`parent`** *(opcional)*: solo para documentación / taxonomía.
  No afecta el entrenamiento.
- **`description`** *(opcional)*: descriptores visuales. Se mezclan
  con los sinónimos al muestrear en train. Útil para clases raras
  cuyo embedding base es débil.

### 3.2 Ejemplos

```yaml
# Clase común, varias lenguas
building:
  synonyms: [building, house, rooftop, edificio, construction]

# Clase específica con padre (vegetación) y descriptores visuales
avocado_tree:
  parent: vegetation
  synonyms: [avocado tree, árbol de aguacate, aguacate]
  description: [dense round canopy, dark green tree]

# Clase muy rara, con descriptores que anclan el embedding
greenhouse:
  synonyms: [greenhouse, invernadero, plastic greenhouse]
  description:
    - transparent rectangular structure
    - agricultural plastic cover
    - row of plastic-roofed structures

# Sub-clase de building, multilingüe
zinc_roof:
  parent: building
  synonyms: [zinc roof, techo de zinc, metal roof, corrugated metal roof]
  description: [reflective gray sheet roof, corrugated metal]
```

## 4. Estrategias por tipo de clase

### 4.1 Clases comunes (building, road, water, vegetation)

- SAM3 base ya las entiende razonablemente. El LoRA te da especialidad
  para tu sensor / resolución / GSD / región.
- 2–4 sinónimos en 2 idiomas suelen bastar.
- Descriptores visuales no son necesarios.

### 4.2 Clases sub-categóricas (avocado_tree vs lemon_tree)

- SAM3 entiende "tree" pero no distingue especies por defecto.
- El LoRA debe aprender la **diferencia visual** entre sub-clases.
- Asegúrate de que tu dataset tenga **ambas** sub-clases si quieres
  separarlas. Si solo entrenas `avocado_tree`, podría detectar también
  los limoneros (no hubo contraste).
- Tip: incluye `description` con rasgos discriminativos
  (`"dense round canopy"` vs `"sparse irregular canopy"`).

### 4.3 Clases raras (`invernadero`, `cancha sintética`)

- SAM3 puede no tener buen embedding base. Es por eso que el LoRA en
  el text encoder está activado por defecto — lo entrena a tu
  vocabulario.
- Usa 4–6 descriptores visuales además de los sinónimos.
- Considera más epochs (60–80) y dataset un poco más grande (≥ 300
  tiles) para estas clases.

### 4.4 Clases negativas / "lo que no es X"

SAM3 no entrena directamente con clases negativas. Si necesitas
distinguir `building` de `non_building`, mejor entrena un solo LoRA de
`building` y trata todo lo demás como fondo. Los tiles que no
contienen `building` pueden ir como tiles "vacíos" (sin anotaciones)
en tu COCO — el modelo aprende a predecir poco / nada.

## 5. Caso de uso: árboles frutales por especie

Quieres distinguir aguacate, limón, mango sobre el mismo huerto.

**Paso 1** — registra las tres clases:

```yaml
avocado_tree:
  parent: vegetation
  synonyms: [avocado tree, árbol de aguacate, aguacate]
  description: [dense round dark green canopy]

lemon_tree:
  parent: vegetation
  synonyms: [lemon tree, árbol de limón, limonero, citrus tree]
  description: [small bright green compact canopy]

mango_tree:
  parent: vegetation
  synonyms: [mango tree, árbol de mango]
  description: [large oval dense canopy, lobed crown]
```

**Paso 2** — un solo COCO con tres categorías:

```json
"categories": [
  {"id": 1, "name": "avocado_tree"},
  {"id": 2, "name": "lemon_tree"},
  {"id": 3, "name": "mango_tree"}
]
```

**Paso 3** — entrenas tres LoRAs independientes:

```bash
for cls in avocado_tree lemon_tree mango_tree; do
  python train_class_lora.py \
    --dataset_root data/huerto \
    --target_class $cls \
    --output_dir outputs/${cls}_lora
done
```

Cada LoRA ve **solo** anotaciones de su clase. Pero al estar etiquetadas
en el mismo COCO, garantizas consistencia geográfica (mismas zonas,
mismo sensor, mismo GSD).

**Paso 4** — inferes con los tres LoRAs y combinas en QGIS.

## 6. Validar que tu clase está reconocida

Antes de entrenar:

```python
from train_sam3_lora_with_categories import SAM3DatasetWithCategories
ds = SAM3DatasetWithCategories(
    root_dir="data/huerto/train",
    target_class="avocado_tree",
)
print(ds._prompt_registry.get("avocado_tree"))
# Debe imprimir {'synonyms': [...], 'description': [...], 'parent': 'vegetation'}
```

Si imprime `None`, está mal el nombre en el YAML.

## 7. Errores típicos al añadir clases

| Síntoma | Causa | Fix |
|---|---|---|
| `target_class='X' not found in COCO categories` | El nombre del YAML no coincide con `categories[].name`. | Renombrar uno de los dos para que coincidan. |
| Warning "Class 'X' not in prompt registry" | No agregaste la entrada al YAML. | Añadirla (opcional pero recomendado). |
| LoRA no aprende a distinguir sub-clases | Solo tienes una sub-clase en el dataset. | Incluir ejemplos de las otras sub-clases (aunque sea como "fondo"). |
| Predicciones razonables en inglés, malas en español | Sinónimos solo en inglés. | Añadir variantes en español al `synonyms`. |
| Predicciones malas con clase muy rara | No hay `description`, embedding base débil. | Añadir 3–5 descriptores visuales. |

## 8. ¿Cuándo conviene entrenar un solo LoRA multi-clase?

Esta plantilla está diseñada para **uno por clase**. Pero si tienes
clases muy correlacionadas (ej. `road` + `pavement` + `street`) y poco
data por clase, puedes:

1. Unificarlas bajo un solo `name` en el COCO.
2. Listar todas las variantes como `synonyms` del único nombre canónico.
3. Entrenar un solo LoRA.

Es válido. La regla "uno por clase" es por diseño, no por restricción.
Si las clases comparten 80% del aspecto visual, mejor un LoRA.
