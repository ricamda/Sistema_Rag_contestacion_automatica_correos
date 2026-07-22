import json
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================================================
# CONFIGURACIÓN
# =====================================================

# Apunta al directorio que contiene todo el Corpus Documental
INPUT_DIR = Path("data/processed_docs/")
# Archivo de salida que contendrá los chunks de ambos dominios
OUTPUT_FILE = Path("chunks_corpus_completo.json")

MAX_CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n\n",
        "\n",
        ". ",
        " "
    ]
)

# =====================================================
# UTILIDADES
# =====================================================

def limpiar_texto(texto):
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    return texto.strip()

def obtener_dominio(nombre_archivo):
    nombre = nombre_archivo.lower()
    
    # Track A: Estudios de Postgrado
    if "postgrado" in nombre:
        return "postgrado"
    
    # Track B: Prácticas Universitarias
    return "practicas"

def extraer_estructura_markdown(texto):
    """
    Coge el primer título que aparezca como título padre del documento.
    Después, cada título siguiente se convierte en un hijo directo del padre,
    junto con su contenido.
    """
    lineas = texto.splitlines()
    titulo_documento = ""
    bloques = []
    titulo_actual = ""
    buffer = []
    patron_titulo = re.compile(r"^(#{1,6})\s+(.*)$")

    def guardar_bloque():
        nonlocal titulo_actual, buffer
        contenido = "\n".join(buffer).strip()
        if titulo_actual or contenido:
            bloques.append({
                "titulo_bloque": titulo_actual.strip(),
                "contenido": contenido
            })
        buffer = []

    for linea in lineas:
        linea_limpia = linea.strip()

        if linea_limpia in ["---", "```"]:
            continue

        match = patron_titulo.match(linea_limpia)
        if match:
            titulo = match.group(2).strip()

            if not titulo_documento:
                titulo_documento = titulo
                continue

            guardar_bloque()
            titulo_actual = titulo
        else:
            buffer.append(linea)

    guardar_bloque()

    if not titulo_documento:
        titulo_documento = "Documento sin título"

    return titulo_documento, bloques

# =====================================================
# GENERACIÓN DE CHUNKS
# =====================================================

def crear_chunks():
    resultados = []
    documento_id = 291

    # Comprobamos que el directorio base existe
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        print(f"❌ Error: No se encuentra el directorio {INPUT_DIR}")
        return

    # Iteramos sobre todos los archivos .md en el directorio
    for archivo in INPUT_DIR.glob("*.md"):
        print(f"Procesando {archivo.name}")

        dominio = obtener_dominio(archivo.name)

        texto = archivo.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        texto = limpiar_texto(texto)
        titulo_doc, bloques = extraer_estructura_markdown(texto)

        # ==========================================
        # PADRE DEL DOCUMENTO
        # ==========================================
        parent_doc_id = str(documento_id)

        resultados.append({
            "contenido": titulo_doc,
            "metadatos": {
                "tipo": "documento",
                "chunk_id": parent_doc_id,
                "parent_chunk": None,
                "Archivo_Origen": archivo.name,
                "dominio": dominio,
                "Seccion": titulo_doc
            }
        })

        # ==========================================
        # HIJOS DIRECTOS DEL DOCUMENTO
        # ==========================================
        hijo_num = 1

        for bloque in bloques:
            titulo_bloque = bloque["titulo_bloque"].strip()
            contenido = bloque["contenido"].strip()

            if not titulo_bloque and not contenido:
                continue

            texto_completo = f"{titulo_bloque}\n\n{contenido}".strip()
            partes = splitter.split_text(texto_completo)

            for parte in partes:
                chunk_id = f"{parent_doc_id}.{hijo_num}"

                texto_vector = f"""
DOCUMENTO: {titulo_doc}

SECCIÓN:
{titulo_bloque}

CONTENIDO:
{parte}
""".strip()

                resultados.append({
                    "contenido": parte,
                    "texto_vector": texto_vector,
                    "metadatos": {
                        "tipo": "contenido",
                        "chunk_id": chunk_id,
                        "parent_chunk": parent_doc_id,
                        "Archivo_Origen": archivo.name,
                        "dominio": dominio,
                        "Seccion": titulo_doc,
                        "Subseccion": titulo_bloque
                    }
                })

                hijo_num += 1

        documento_id += 1

    # Crear el directorio de salida si no existe
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            resultados,
            f,
            ensure_ascii=False,
            indent=2
        )

    documentos_procesados = documento_id - 291
    print(f"\n✅ Generados {len(resultados)} chunks a partir de {documentos_procesados} documentos.")
    print(f"📁 Guardado en {OUTPUT_FILE}")

if __name__ == "__main__":
    crear_chunks()