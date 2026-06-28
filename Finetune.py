"""
Fine-tuning de ViT-tiny
"""

import os
import json
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image


# CONFIGURACION
CONFIG = {
    "model_id":       "WinKawaks/vit-tiny-patch16-224",
    "dataset_dir":    "./dataset_plantid",
    "output_dir":     "./plantnet_vit_finetuned_v5_unfrozen",
    "num_epochs":     20,
    "batch_size":     16,
    "lr_cabeza":      2e-4,   # cabeza clasificadora
    "lr_backbone":    2e-5,   # ultimas capas del ViT (10x mas bajo)
    "val_split":      0.2,
    "seed":           42,
    "early_stop":     5,      # parar si no mejora en 5 epochs
    "bloques_descongelar": ["layers.10", "layers.11", "classifier"],
}


# TRANSFORMS
TRANSFORM_TRAIN = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.1),
    T.RandomRotation(degrees=20),
    T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
    T.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
    T.RandomGrayscale(p=0.05),
])

TRANSFORM_VAL = None


# DATASET
class PlantDatasetBase(Dataset):
    """Lista todas las muestras sin aplicar transforms (para hacer el split primero)."""
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

        print(f"Dataset: {len(self.muestras)} imagenes | {len(label2idx)} clases")

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, i):
        return self.muestras[i]  # (ruta, label) — transforms se aplican en el wrapper


class SubsetConTransform(Dataset):
    """Aplica transform especifico a un subset del dataset base."""
    def __init__(self, muestras, image_processor, transform=None):
        self.muestras        = muestras
        self.image_processor = image_processor
        self.transform       = transform

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, i):
        ruta, label = self.muestras[i]
        img = Image.open(ruta).convert("RGB")
        if self.transform:
            img = self.transform(img)
        inputs = self.image_processor(images=img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), torch.tensor(label, dtype=torch.long)


# ENTRENAMIENTO
def entrenar_local(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    # Cargar mapa de etiquetas
    with open(f"{config['dataset_dir']}/label_map.json") as f:
        idx2label = json.load(f)
    label2idx   = {v: int(k) for k, v in idx2label.items()}
    num_classes = len(label2idx)
    print(f"Clases: {num_classes}")

    # Procesador de imagenes
    image_processor = AutoImageProcessor.from_pretrained(config["model_id"])

    # Dataset base y split train/val
    dataset_base = PlantDatasetBase(config["dataset_dir"], label2idx)
    torch.manual_seed(config["seed"])
    n_val   = int(config["val_split"] * len(dataset_base))
    n_train = len(dataset_base) - n_val
    gen     = torch.Generator().manual_seed(config["seed"])
    train_sub, val_sub = random_split(dataset_base, [n_train, n_val], generator=gen)

    train_muestras = [dataset_base.muestras[i] for i in train_sub.indices]
    val_muestras   = [dataset_base.muestras[i] for i in val_sub.indices]

    train_dataset = SubsetConTransform(train_muestras, image_processor, TRANSFORM_TRAIN)
    val_dataset   = SubsetConTransform(val_muestras,   image_processor, TRANSFORM_VAL)

    print(f"Train: {len(train_dataset)} imagenes (con augmentation)")
    print(f"Val:   {len(val_dataset)} imagenes (sin augmentation)\n")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    # Modelo con cabeza nueva
    modelo = AutoModelForImageClassification.from_pretrained(
        config["model_id"],
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    # Congelar TODO el backbone primero
    for param in modelo.parameters():
        param.requires_grad = False

    # Descongelar solo los bloques seleccionados + cabeza
    for nombre, param in modelo.named_parameters():
        if any(bloque in nombre for bloque in config["bloques_descongelar"]):
            param.requires_grad = True

    # Reporte de parametros entrenables
    params_total   = sum(p.numel() for p in modelo.parameters())
    params_entrena = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"Parametros totales:     {params_total:,}")
    print(f"Parametros entrenables: {params_entrena:,} "
          f"({params_entrena/params_total*100:.1f}% del modelo)")

    # VALIDACION DE SEGURIDAD
    backbone_descongelado = [n for n, p in modelo.named_parameters()
                              if p.requires_grad and "classifier" not in n]
    if len(backbone_descongelado) == 0:
        nombres_ejemplo = [n for n, _ in modelo.named_parameters()][:5]
        raise ValueError(
            "El filtro 'bloques_descongelar' no coincidio con ningun "
            "parametro del backbone (solo se descongelo la cabeza).\n"
            f"Bloques buscados: {config['bloques_descongelar']}\n"
            f"Ejemplo de nombres reales en el modelo: {nombres_ejemplo}\n"
            "Revisa los nombres con named_parameters() y ajusta el CONFIG."
        )
    print(f"Parametros del backbone descongelados: {len(backbone_descongelado)} tensores\n")

    modelo.to(device)

    # Optimizador con LR diferenciado
    params_backbone = [p for n, p in modelo.named_parameters()
                       if p.requires_grad and "classifier" not in n]
    params_cabeza   = [p for n, p in modelo.named_parameters()
                       if p.requires_grad and "classifier" in n]

    optimizer = torch.optim.AdamW([
        {"params": params_backbone, "lr": config["lr_backbone"]},
        {"params": params_cabeza,   "lr": config["lr_cabeza"]},
    ], weight_decay=0.01)

    # Scheduler: reduce LR x0.5 si Val Acc no mejora en 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    criterion = nn.CrossEntropyLoss()

    mejor_val_acc  = 0.0
    epochs_sin_mejora = 0
    os.makedirs(config["output_dir"], exist_ok=True)

    print("-" * 60)
    print("ENTRENAMIENTO")
    print("-" * 60)

    for epoch in range(config["num_epochs"]):

        # TRAIN
        modelo.train()
        perdida_acum = correctas = total = 0

        for batch_idx, (pixel_values, labels) in enumerate(train_loader):
            pixel_values = pixel_values.to(device)
            labels       = labels.to(device)

            optimizer.zero_grad()
            logits = modelo(pixel_values=pixel_values).logits
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            correctas    += (logits.argmax(dim=-1) == labels).sum().item()
            perdida_acum += loss.item()
            total        += labels.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:02d}/{config['num_epochs']} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {perdida_acum/(batch_idx+1):.4f} | "
                      f"Train Acc: {correctas/total*100:.1f}%")

        train_acc  = correctas / total * 100
        train_loss = perdida_acum / len(train_loader)

        # VAL
        modelo.eval()
        val_correctas = val_total = 0
        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values = pixel_values.to(device)
                labels       = labels.to(device)
                logits       = modelo(pixel_values=pixel_values).logits
                val_correctas += (logits.argmax(dim=-1) == labels).sum().item()
                val_total     += labels.size(0)

        val_acc = val_correctas / val_total * 100 if val_total > 0 else 0.0

        print(f"\nEpoch {epoch+1:02d} -- "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Val Acc: {val_acc:.1f}%", end="")

        # Scheduler paso
        scheduler.step(val_acc)

        # Guardar si mejora
        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc
            epochs_sin_mejora = 0
            modelo.save_pretrained(config["output_dir"])
            image_processor.save_pretrained(config["output_dir"])
            shutil.copy(f"{config['dataset_dir']}/label_map.json",
                        f"{config['output_dir']}/label_map.json")
            print(" -> Mejor modelo guardado")
        else:
            epochs_sin_mejora += 1
            print(f" (sin mejora: {epochs_sin_mejora}/{config['early_stop']})")

        # Early stopping
        if epochs_sin_mejora >= config["early_stop"]:
            print(f"\nEarly stopping en epoch {epoch+1} -- "
                  f"no hubo mejora en {config['early_stop']} epochs consecutivos.")
            break

        print()

    print(f"\n{'='*60}")
    print(f"Mejor Val Acc: {mejor_val_acc:.1f}%")
    print(f"Modelo guardado en: {config['output_dir']}/")


# MAIN
if __name__ == "__main__":
    entrenar_local(CONFIG)