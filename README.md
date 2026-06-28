# 🌿 PlantBot — Entrenamiento del modelo ViT

Repositorio con los scripts que generan el modelo de visión usado en [PlantBot](https://huggingface.co/spaces/RenSASO/plantbot), un chatbot que identifica plantas mexicanas a partir de fotografías.

El modelo resultante es un **ViT-tiny fine-tuneado** capaz de clasificar 99 especies de plantas, con énfasis en flora mexicana, plantas medicinales y especies tóxicas o de riesgo.

---

## Archivos

| Archivo | Descripción |
|---|---|
| `Dataset.py` | Descarga imágenes de iNaturalist para las 99 especies |
| `Finetune.py` | Fine-tuning del modelo ViT-tiny sobre el dataset descargado |
| `confusion.py` | Evaluación del modelo: matriz de confusión y detección de errores peligrosos |

---

## Flujo de uso

### 1. Descargar el dataset

```bash
python Dataset.py
```

Descarga ~80 imágenes por especie desde la API pública de iNaturalist (solo observaciones verificadas por expertos, `quality_grade=research`). Las imágenes se redimensionan a 224×224 y se organizan en carpetas por especie:

```
dataset_plantid/
    Aloe_vera/
    Datura_stramonium/
    Nerium_oleander/
    ...
```

Al terminar genera automáticamente `dataset_plantid/label_map.json` con el mapeo `índice → nombre científico`.

### 2. Entrenar el modelo

```bash
python Finetune.py
```

Hace fine-tuning de [`WinKawaks/vit-tiny-patch16-224`](https://huggingface.co/WinKawaks/vit-tiny-patch16-224) con las siguientes decisiones de diseño:

- **Congelamiento parcial**: solo se descongelan las últimas 2 capas del backbone (`layers.10`, `layers.11`) más la cabeza clasificadora, para evitar sobreajuste con un dataset pequeño.
- **Learning rate diferenciado**: la cabeza entrena a `2e-4` y el backbone descongelado a `2e-5` (10× más lento).
- **Data augmentation**: volteo horizontal/vertical, rotación ±20°, jitter de color y recorte aleatorio, aplicado solo al set de entrenamiento.
- **Early stopping**: detiene el entrenamiento si la accuracy de validación no mejora en 5 epochs consecutivos.
- **Scheduler**: reduce el LR a la mitad si no hay mejora en 3 epochs.

El mejor checkpoint se guarda en `./plantnet_vit_finetuned/` con los 4 archivos que necesita la app:

```
plantnet_vit_finetuned/
    model.safetensors       ← pesos del modelo
    config.json             ← arquitectura
    preprocessor_config.json← cómo normalizar las imágenes
    label_map.json          ← índice → nombre científico
```

### 3. Evaluar el modelo

```bash
python confusion.py
```

Evalúa el checkpoint sobre el set de validación (mismo split que en entrenamiento) y genera tres salidas en `./evaluacion/`:

- `matriz_confusion.csv` — matriz completa 99×99
- `heatmap_top_errores.png` — heatmap de las 20 especies con más errores
- `reporte_por_especie.txt` — precisión, recall y F1 por especie
- `confusiones_peligrosas.csv` — pares donde el modelo confunde una especie tóxica con una segura (o viceversa), ordenados por frecuencia

La detección de **confusiones peligrosas** es la parte más importante de la evaluación: cada especie tiene asignado un nivel de riesgo (de `Seguro` a `Muy alto`) y el script alerta cuando el modelo comete un error que saltea 3 o más niveles de riesgo, por ejemplo confundir *Nerium oleander* (Muy alto) con *Lavandula angustifolia* (Seguro).

---

## Requisitos

```
torch
torchvision
transformers
Pillow
scikit-learn
matplotlib
requests
```

Instalar con:

```bash
pip install torch torchvision transformers Pillow scikit-learn matplotlib requests
```

Se recomienda GPU para el entrenamiento. En CPU el fine-tuning es lento (~10 min/epoch con el dataset completo).

---

## Especies cubiertas

El modelo clasifica 99 especies organizadas en tres grupos:

- **Tóxicas / ornamentales de riesgo** (34 especies): *Nerium oleander*, *Datura stramonium*, *Ricinus communis*, *Brugmansia suaveolens*, entre otras.
- **Medicinales con riesgo variable** (35 especies): *Aloe vera*, *Artemisia absinthium*, *Ruta graveolens*, *Hypericum perforatum*, entre otras.
- **Flora nativa / uso alternativo** (30 especies): *Agave tequilana*, *Opuntia ficus-indica*, *Dahlia pinnata*, *Taxodium mucronatum*, entre otras.

La lista completa con niveles de riesgo está documentada en `confusion.py` en el diccionario `RIESGO_POR_GENERO`.

---

## Créditos

- Imágenes: [iNaturalist](https://www.inaturalist.org/) (licencia abierta, solo observaciones con `quality_grade=research`)
- Modelo base: [`WinKawaks/vit-tiny-patch16-224`](https://huggingface.co/WinKawaks/vit-tiny-patch16-224)
- App de demostración: [PlantBot en Hugging Face Spaces](https://huggingface.co/spaces/RenSASO/plantbot)
