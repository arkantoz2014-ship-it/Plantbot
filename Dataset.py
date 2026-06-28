"""
Descarga imagenes de tus 100 plantas directamente desde iNaturalist.

Resultado:
    ./dataset_plantid/
        Dieffenbachia_seguine/
        Nerium_oleander/
        ...  (una carpeta por especie, nombrada con el nombre ORIGINAL
              sin importar que alias se haya usado para buscarla)
"""

import os
import time
import json
import requests
from PIL import Image
from io import BytesIO

# Lista completa de las 100 plantas (nombres "oficiales" para las carpetas/label_map)
PLANTAS_PLANTID = [
    # Toxicas / Ornamentales de riesgo
    "Dieffenbachia seguine",
    "Epipremnum aureum",
    "Nerium oleander",
    "Spathiphyllum wallisii",
    "Euphorbia pulcherrima",
    "Philodendron hederaceum",
    "Caladium bicolor",
    "Colocasia esculenta",
    "Ricinus communis",
    "Datura stramonium",
    "Brugmansia suaveolens",
    "Thevetia peruviana",
    "Abrus precatorius",
    "Cycas revoluta",
    "Solanum pseudocapsicum",
    "Lantana camara",
    "Brunfelsia pauciflora",
    "Hedera helix",
    "Dracaena fragrans",
    "Aglaonema commutatum",
    "Asparagus densiflorus",
    "Crassula ovata",
    "Kalanchoe blossfeldiana",
    "Zamioculcas zamiifolia",
    "Schefflera actinophylla",
    "Cestrum nocturnum",
    "Allamanda cathartica",
    "Plumeria rubra",
    "Strelitzia reginae",
    "Agave americana",
    "Ficus benjamina",
    "Hippeastrum puniceum",
    "Convallaria majalis",
    "Lilium candidum",
    # Medicinales con riesgo
    "Aloe vera",
    "Peumus boldus",
    "Valeriana officinalis",
    "Gnaphalium obtusifolium",
    "Dysphania ambrosioides",
    "Argemone mexicana",
    "Ruta graveolens",
    "Matricaria chamomilla",
    "Mentha spicata",
    "Tilia platyphyllos",
    "Foeniculum vulgare",
    "Equisetum arvense",
    "Cinnamomum verum",
    "Eucalyptus globulus",
    "Tagetes lucida",
    "Annona muricata",
    "Calea zacatechichi",
    "Turnera diffusa",
    "Passiflora incarnata",
    "Salvia officinalis",
    "Artemisia absinthium",
    "Senna alexandrina",
    "Symphytum officinale",
    "Larrea tridentata",
    "Persea americana",
    "Citrus aurantium",
    "Hypericum perforatum",
    "Piper auritum",
    "Handroanthus impetiginosus",
    "Cnidoscolus chayamansa",
    "Jatropha curcas",
    "Momordica charantia",
    "Lippia graveolens",
    "Zingiber officinale",
    "Heterotheca inuloides",
    # Flora nativa / uso alternativo
    "Agave tequilana",
    "Opuntia ficus-indica",
    "Bougainvillea glabra",
    "Tagetes erecta",
    "Tropaeolum majus",
    "Ocimum basilicum",
    "Lavandula angustifolia",
    "Cymbopogon nardus",
    "Mirabilis jalapa",
    "Dahlia pinnata",
    "Echinocactus platyacanthus",
    "Yucca filifera",
    "Fouquieria splendens",
    "Bursera graveolens",
    "Salvia mexicana",
    "Cosmos bipinnatus",
    "Tigridia pavonia",
    "Echeveria elegans",
    "Carnegiea gigantea",
    "Pachycereus pringlei",
    "Quercus robur",
    "Taxodium mucronatum",
    "Erythrina americana",
    "Calliandra surinamensis",
    "Heliconia rostrata",
    "Ipomoea purpurea",
    "Crescentia cujete",
    "Bauhinia variegata",
    "Yucca elephantipes",
    "Ceiba pentandra",
]

ALIAS_BUSQUEDA = {
    "Schefflera actinophylla": "Heptapleurum arboricola",
    "Citrus aurantium":        "Citrus sinensis",
    "Cnidoscolus chayamansa":  "Cnidoscolus aconitifolius",
}

# Configuracion
IMAGENES_POR_ESPECIE = 80   # imagenes a descargar por planta
DIRECTORIO_SALIDA    = "./dataset_plantid"
QUALITY_GRADE        = "research"  # solo obs. verificadas por expertos en iNat

os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)


def buscar_taxon_id(nombre_cientifico: str, intentos: int = 3) -> int | None:
    """Busca el taxon_id en iNaturalist por nombre cientifico.
    Tolerante: reintenta en fallos de red, no exige rank=species
    (algunas especies aparecen como hybrid o con autoria distinta),
    y compara contra nombre comun ademas del nombre cientifico."""
    genero = nombre_cientifico.lower().split()[0]

    for intento in range(1, intentos + 1):
        try:
            r = requests.get(
                "https://api.inaturalist.org/v1/taxa",
                params={"q": nombre_cientifico, "per_page": 5},
                timeout=10,
            )
            if r.status_code != 200:
                time.sleep(1.5)
                continue

            resultados = r.json().get("results", [])
            if not resultados:
                return None

            for res in resultados:
                nombre_inat = res.get("name", "").lower()
                comun_inat  = (res.get("preferred_common_name") or "").lower()
                if genero in nombre_inat or genero in comun_inat:
                    return res["id"]

            # ultimo recurso: usar el primer resultado aunque no matchee genero
            return resultados[0]["id"]

        except Exception:
            time.sleep(1.5)

    return None


def descargar_imagenes(nombre_carpeta: str, taxon_id: int, n: int) -> int:
    """Descarga n imagenes de una especie desde iNaturalist. Retorna cuantas se guardaron."""
    carpeta = os.path.join(DIRECTORIO_SALIDA, nombre_carpeta.replace(" ", "_"))
    os.makedirs(carpeta, exist_ok=True)

    existentes = len([f for f in os.listdir(carpeta) if f.endswith(".jpg")])
    if existentes >= n:
        return existentes

    try:
        r = requests.get(
            "https://api.inaturalist.org/v1/observations",
            params={
                "taxon_id":     taxon_id,
                "quality_grade": QUALITY_GRADE,
                "photos":       "true",
                "per_page":     n * 2,
                "order_by":     "votes",
            },
            timeout=10,
        )
        observaciones = r.json().get("results", [])
    except Exception as e:
        print(f"    Error al consultar observaciones: {e}")
        return 0

    guardadas = 0
    for obs in observaciones:
        if guardadas >= n:
            break
        fotos = obs.get("photos", [])
        if not fotos:
            continue
        url = fotos[0].get("url", "").replace("square", "medium")
        if not url:
            continue
        try:
            img_r = requests.get(url, timeout=10)
            img   = Image.open(BytesIO(img_r.content)).convert("RGB")
            img   = img.resize((224, 224))  # tamano estandar para ViT
            ruta  = os.path.join(carpeta, f"{taxon_id}_{obs['id']}.jpg")
            img.save(ruta, "JPEG", quality=85)
            guardadas += 1
            time.sleep(0.1)
        except Exception:
            continue

    return guardadas


# Ejecucion principal
print(f"Descargando imagenes para {len(PLANTAS_PLANTID)} especies...")
print(f"Destino: {DIRECTORIO_SALIDA}/\n")

resumen_ok  = []
resumen_mal = []

for i, nombre in enumerate(PLANTAS_PLANTID, 1):
    alias = ALIAS_BUSQUEDA.get(nombre, nombre)
    etiqueta = f"{nombre} (via {alias})" if alias != nombre else nombre
    print(f"[{i:03d}/{len(PLANTAS_PLANTID)}] {etiqueta}", end=" ... ")

    taxon_id = buscar_taxon_id(alias)
    if not taxon_id:
        print("\u274c No encontrada en iNaturalist")
        resumen_mal.append(nombre)
        time.sleep(0.3)
        continue

    n_guardadas = descargar_imagenes(nombre, taxon_id, IMAGENES_POR_ESPECIE)
    if n_guardadas > 0:
        print(f"\u2705 {n_guardadas} imagenes (taxon {taxon_id})")
        resumen_ok.append((nombre, n_guardadas))
    else:
        print(f"\u26a0\ufe0f  Taxon {taxon_id} encontrado pero sin imagenes descargables")
        resumen_mal.append(nombre)

    time.sleep(0.4)  # respetar rate limit de iNaturalist

# Resumen final
total_imagenes = sum(n for _, n in resumen_ok)
print(f"\n{'='*55}")
print(f"RESUMEN FINAL")
print(f"  Especies descargadas: {len(resumen_ok)}/{len(PLANTAS_PLANTID)}")
print(f"  No encontradas/sin fotos: {len(resumen_mal)}")
print(f"  Total imagenes: {total_imagenes}")
print(f"  Directorio: {DIRECTORIO_SALIDA}/")
if resumen_mal:
    print(f"\n  Especies no descargadas:")
    for nombre in resumen_mal:
        print(f"    - {nombre}")

# Construir label2idx e idx2label para el entrenamiento
label2idx = {}
idx2label = {}

carpetas = sorted(os.listdir(DIRECTORIO_SALIDA))
carpetas = [c for c in carpetas if os.path.isdir(os.path.join(DIRECTORIO_SALIDA, c))]
for idx, carpeta in enumerate(carpetas):
    nombre_cientifico = carpeta.replace("_", " ")
    label2idx[nombre_cientifico] = idx
    idx2label[idx] = nombre_cientifico

with open(f"{DIRECTORIO_SALIDA}/label_map.json", "w") as f:
    json.dump(idx2label, f, ensure_ascii=False, indent=2)

print(f"\n  label_map.json guardado con {len(idx2label)} clases.")
print(f"  Usa este valor en CONFIG:")
print(f'    "num_classes": {len(idx2label)},')