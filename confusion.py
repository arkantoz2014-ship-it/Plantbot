"""
Evaluacion del modelo entrenado: matriz de confusion + analisis de
"confusiones peligrosas" (el modelo confunde una especie toxica con
una segura, o viceversa).
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


# CONFIGURACION (debe coincidir con el entrenamiento)
CONFIG = {
    "dataset_dir":  "./dataset_plantid",
    "model_dir":    "./plantnet_vit_finetuned_v5_unfrozen",  # checkpoint a evaluar
    "val_split":    0.2,
    "seed":         42,
    "batch_size":   16,
}

# MAPA DE RIESGO POR GENERO
# Resumen de la tabla de 100 plantas, simplificado por genero (primera
# palabra del nombre cientifico) para poder cruzarlo facilmente contra
# las especies reales del dataset. Ajusta si la tabla define algo distinto.
RIESGO_POR_GENERO = {
    # Toxicas / ornamentales de riesgo (alto-muy alto)
    "dieffenbachia": "Alto", "epipremnum": "Medio", "nerium": "Muy alto",
    "spathiphyllum": "Medio", "euphorbia": "Bajo-Medio", "philodendron": "Medio",
    "caladium": "Medio", "colocasia": "Medio", "ricinus": "Muy alto",
    "datura": "Muy alto", "brugmansia": "Muy alto", "thevetia": "Muy alto",
    "abrus": "Muy alto", "cycas": "Alto", "solanum": "Medio-Alto",
    "lantana": "Alto", "brunfelsia": "Alto", "hedera": "Medio",
    "dracaena": "Bajo-Medio", "aglaonema": "Medio", "asparagus": "Medio",
    "crassula": "Bajo", "kalanchoe": "Medio", "zamioculcas": "Medio",
    "schefflera": "Medio", "heptapleurum": "Medio", "cestrum": "Alto",
    "allamanda": "Medio-Alto", "plumeria": "Bajo", "strelitzia": "Bajo-Medio",
    "ficus": "Bajo", "hippeastrum": "Medio-Alto", "convallaria": "Muy alto",
    "lilium": "Alto",
    # Medicinales con riesgo (bajo-alto, variable)
    "aloe": "Bajo", "peumus": "Medio-Alto", "valeriana": "Medio",
    "gnaphalium": "Bajo-Medio", "dysphania": "Alto", "argemone": "Alto",
    "ruta": "Alto", "matricaria": "Bajo", "mentha": "Bajo",
    "tilia": "Bajo-Medio", "foeniculum": "Bajo-Medio", "equisetum": "Medio",
    "cinnamomum": "Medio", "eucalyptus": "Medio-Alto", "tagetes": "Seguro",
    "annona": "Medio-Alto", "calea": "Medio-Alto", "turnera": "Medio",
    "passiflora": "Bajo-Medio", "salvia": "Medio", "artemisia": "Alto",
    "senna": "Medio-Alto", "symphytum": "Alto", "larrea": "Alto",
    "persea": "Medio", "citrus": "Medio", "hypericum": "Alto",
    "piper": "Medio", "handroanthus": "Medio", "cnidoscolus": "Medio",
    "jatropha": "Alto", "momordica": "Medio", "lippia": "Bajo-Medio",
    "zingiber": "Bajo-Medio", "heterotheca": "Medio-Alto",
    # Flora nativa / uso alternativo (en su mayoria seguras)
    "bougainvillea": "Seguro", "tropaeolum": "Seguro", "ocimum": "Seguro",
    "lavandula": "Seguro", "cymbopogon": "Seguro", "mirabilis": "Medio",
    "dahlia": "Seguro", "echinocactus": "Seguro", "yucca": "Seguro",
    "fouquieria": "Seguro", "bursera": "Seguro", "cosmos": "Seguro",
    "tigridia": "Seguro", "echeveria": "Seguro", "carnegiea": "Seguro",
    "pachycereus": "Seguro", "quercus": "Seguro", "taxodium": "Seguro",
    "erythrina": "Alto", "calliandra": "Seguro", "heliconia": "Seguro",
    "ipomoea": "Medio-Alto", "crescentia": "Seguro", "bauhinia": "Seguro",
    "ceiba": "Seguro", "agave": "Bajo-Medio",
}

# Orden de severidad para detectar "saltos" grandes entre niveles
ORDEN_RIESGO = ["Seguro", "Bajo", "Bajo-Medio", "Medio", "Medio-Alto", "Alto", "Muy alto"]


def riesgo_de(nombre_cientifico: str) -> str:
    genero = nombre_cientifico.lower().split()[0]
    return RIESGO_POR_GENERO.get(genero, "Desconocido")


def distancia_riesgo(r1: str, r2: str) -> int:
    """Distancia en niveles de severidad. -1 si alguno es 'Desconocido'."""
    if r1 not in ORDEN_RIESGO or r2 not in ORDEN_RIESGO:
        return -1
    return abs(ORDEN_RIESGO.index(r1) - ORDEN_RIESGO.index(r2))


# DATASET (igual al de entrenamiento)
class PlantDatasetBase(Dataset):
    def __init__(self, dataset_dir: str, label2idx: dict):
        self.label2idx = label2idx
        self.muestras  = []
        for carpeta in sorted(os.listdir(dataset_dir)):
            ruta_carpeta = os.path.join(dataset_dir, carpeta)
            if not os.path.isdir(ruta_carpeta):
                continue
            nombre = carpeta.replace("_", " ")
            if nombre not in label2idx:
                continue
            idx = label2idx[nombre]
            for archivo in os.listdir(ruta_carpeta):
                if archivo.endswith(".jpg"):
                    self.muestras.append((os.path.join(ruta_carpeta, archivo), idx))

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, i):
        return self.muestras[i]


class ValDataset(Dataset):
    def __init__(self, muestras, image_processor):
        self.muestras        = muestras
        self.image_processor = image_processor

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, i):
        ruta, label = self.muestras[i]
        img = Image.open(ruta).convert("RGB")
        inputs = self.image_processor(images=img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), torch.tensor(label, dtype=torch.long)


def evaluar(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    with open(f"{config['dataset_dir']}/label_map.json") as f:
        idx2label = json.load(f)
    label2idx   = {v: int(k) for k, v in idx2label.items()}
    idx2label   = {int(k): v for k, v in idx2label.items()}
    num_classes = len(label2idx)

    image_processor = AutoImageProcessor.from_pretrained(config["model_dir"])
    modelo = AutoModelForImageClassification.from_pretrained(config["model_dir"])
    modelo.to(device)
    modelo.eval()

    # Mismo split que en entrenamiento, para evaluar solo sobre val
    dataset_base = PlantDatasetBase(config["dataset_dir"], label2idx)
    torch.manual_seed(config["seed"])
    n_val   = int(config["val_split"] * len(dataset_base))
    n_train = len(dataset_base) - n_val
    gen     = torch.Generator().manual_seed(config["seed"])
    _, val_sub = random_split(dataset_base, [n_train, n_val], generator=gen)
    val_muestras = [dataset_base.muestras[i] for i in val_sub.indices]

    val_dataset = ValDataset(val_muestras, image_processor)
    val_loader  = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    print(f"Evaluando sobre {len(val_dataset)} imagenes de validacion...\n")

    y_true, y_pred = [], []
    with torch.no_grad():
        for pixel_values, labels in val_loader:
            pixel_values = pixel_values.to(device)
            logits = modelo(pixel_values=pixel_values).logits
            preds  = logits.argmax(dim=-1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = (y_true == y_pred).mean() * 100
    print(f"Accuracy global en validacion: {acc:.1f}%\n")

    # Matriz de confusion completa (99x99) guardada como CSV para inspeccion
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    nombres = [idx2label[i] for i in range(num_classes)]

    os.makedirs("./evaluacion", exist_ok=True)
    import csv
    with open("./evaluacion/matriz_confusion.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + nombres)
        for i, fila in enumerate(cm):
            writer.writerow([nombres[i]] + fila.tolist())
    print("Matriz de confusion completa guardada en ./evaluacion/matriz_confusion.csv")

    # Top pares mas confundidos
    pares = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                pares.append((nombres[i], nombres[j], int(cm[i, j])))
    pares.sort(key=lambda x: -x[2])

    print("\n=== TOP 20 CONFUSIONES MAS FRECUENTES ===")
    print(f"{'Real':<30} {'Predicho':<30} {'Veces':<6} {'Riesgo real':<12} {'Riesgo pred':<12} {'Alerta'}")
    print("-" * 110)
    for real, pred, veces in pares[:20]:
        r_real = riesgo_de(real)
        r_pred = riesgo_de(pred)
        dist   = distancia_riesgo(r_real, r_pred)
        alerta = "ALERTA: PELIGROSA" if dist >= 3 else ("leve" if dist >= 1 else "")
        print(f"{real:<30} {pred:<30} {veces:<6} {r_real:<12} {r_pred:<12} {alerta}")

    # Filtrar SOLO las confusiones peligrosas
    peligrosas = [(r, p, v, riesgo_de(r), riesgo_de(p)) for r, p, v in pares
                  if distancia_riesgo(riesgo_de(r), riesgo_de(p)) >= 3]
    peligrosas.sort(key=lambda x: -x[2])

    print(f"\n=== CONFUSIONES PELIGROSAS (salto de >=3 niveles de riesgo) ===")
    if not peligrosas:
        print("Ninguna detectada. Buena senal: el modelo no confunde "
              "niveles de riesgo muy distintos.")
    else:
        print(f"{'Real':<30} {'Predicho':<30} {'Veces':<6} {'Riesgo real':<12} {'Riesgo pred':<12}")
        print("-" * 100)
        for real, pred, veces, r_real, r_pred in peligrosas:
            print(f"{real:<30} {pred:<30} {veces:<6} {r_real:<12} {r_pred:<12}")

    with open("./evaluacion/confusiones_peligrosas.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["especie_real", "especie_predicha", "veces",
                          "riesgo_real", "riesgo_predicho"])
        for real, pred, veces, r_real, r_pred in peligrosas:
            writer.writerow([real, pred, veces, r_real, r_pred])

    # Heatmap solo de las clases con mas errores (legible, no 99x99 completo)
    errores_por_clase = cm.sum(axis=1) - np.diag(cm)
    top_n = 20
    indices_top = np.argsort(-errores_por_clase)[:top_n]

    cm_sub = cm[np.ix_(indices_top, indices_top)]
    nombres_sub = [nombres[i] for i in indices_top]

    plt.figure(figsize=(14, 12))
    plt.imshow(cm_sub, cmap="Reds")
    plt.colorbar(label="Numero de imagenes")
    plt.xticks(range(top_n), nombres_sub, rotation=90, fontsize=8)
    plt.yticks(range(top_n), nombres_sub, fontsize=8)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusion -- Top {top_n} especies con mas errores")
    plt.tight_layout()
    plt.savefig("./evaluacion/heatmap_top_errores.png", dpi=150)
    print(f"\nHeatmap de las {top_n} especies con mas errores guardado en "
          f"./evaluacion/heatmap_top_errores.png")

    # Reporte completo por clase (precision/recall/f1)
    reporte = classification_report(y_true, y_pred, target_names=nombres,
                                     labels=list(range(num_classes)),
                                     zero_division=0)
    with open("./evaluacion/reporte_por_especie.txt", "w", encoding="utf-8") as f:
        f.write(reporte)
    print("Reporte de precision/recall por especie guardado en "
          "./evaluacion/reporte_por_especie.txt")


if __name__ == "__main__":
    evaluar(CONFIG)