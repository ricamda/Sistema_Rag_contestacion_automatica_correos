"""
Generación de embeddings híbridos e indexación local en Qdrant.

Este script realiza dos tareas principales:

1. Generar representaciones densas, dispersas y ColBERT para los chunks
   documentales mediante el modelo BAAI/bge-m3.
2. Crear una colección local de Qdrant con los vectores densos y dispersos,
   manteniendo en el payload toda la información original de cada chunk.

El script está pensado como un proceso de preparación previo al arranque de
la API principal. No debe ejecutarse cada vez que se inicia la aplicación,
ya que recalcula todos los embeddings y reconstruye completamente Qdrant.
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

MODEL_NAME = "BAAI/bge-m3"

CHUNKS_INPUT_PATH = Path("./chunks_unificados.json")
EMBEDDINGS_DIR = Path("./embeddings_export")

FILTERED_CHUNKS_PATH = EMBEDDINGS_DIR / "chunks_filtrados.json"
DENSE_VECTORS_PATH = EMBEDDINGS_DIR / "dense_vecs.npy"
SPARSE_VECTORS_PATH = EMBEDDINGS_DIR / "lexical_weights.pkl"
COLBERT_VECTORS_PATH = EMBEDDINGS_DIR / "colbert_vecs.pkl"

QDRANT_PATH = Path("./qdrant_hibrido")
COLLECTION_NAME = "uv_practicas"

BATCH_SIZE_EMBEDDINGS = 2
MAX_SEQUENCE_LENGTH = 1024
BATCH_SIZE_QDRANT = 64


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CARGA Y FILTRADO DE CHUNKS
# ---------------------------------------------------------------------------

def cargar_chunks(ruta: Path) -> list[dict[str, Any]]:
    """Carga los chunks documentales desde un archivo JSON."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo de chunks: {ruta}")

    with ruta.open("r", encoding="utf-8") as file:
        datos = json.load(file)

    if not isinstance(datos, list):
        raise ValueError("El archivo de chunks debe contener una lista JSON.")

    return datos


def es_padre_solo_titulo(chunk: dict[str, Any]) -> bool:
    """
    Indica si un chunk representa únicamente un nodo padre sin texto vectorizable.

    Estos nodos se excluyen porque sirven para conservar la jerarquía documental,
    pero no contienen contenido útil para la recuperación semántica.
    """
    metadatos = chunk.get("metadatos", {})
    return metadatos.get("parent_chunk") is None and "texto_vector" not in chunk


def filtrar_chunks_vectorizables(
    chunks_todos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Elimina los chunks padre que contienen únicamente un título."""
    chunks_filtrados = [
        chunk for chunk in chunks_todos if not es_padre_solo_titulo(chunk)
    ]

    logger.info("Chunks totales: %s", len(chunks_todos))
    logger.info(
        "Chunks excluidos por contener solo un título: %s",
        len(chunks_todos) - len(chunks_filtrados),
    )
    logger.info("Chunks que se vectorizarán: %s", len(chunks_filtrados))

    if not chunks_filtrados:
        raise ValueError("No hay chunks válidos para generar embeddings.")

    return chunks_filtrados


def obtener_textos_vectorizacion(
    chunks: list[dict[str, Any]],
) -> list[str]:
    """
    Obtiene el texto utilizado por el modelo de embeddings.

    Se prioriza el campo ``texto_vector`` cuando existe. En caso contrario,
    se utiliza ``contenido``.
    """
    textos: list[str] = []

    for indice, chunk in enumerate(chunks):
        texto = chunk.get("texto_vector") or chunk.get("contenido")

        if not isinstance(texto, str) or not texto.strip():
            raise ValueError(
                f"El chunk situado en la posición {indice} no contiene texto válido."
            )

        textos.append(texto.strip())

    return textos


# ---------------------------------------------------------------------------
# GENERACIÓN Y GUARDADO DE EMBEDDINGS
# ---------------------------------------------------------------------------

def cargar_modelo_embeddings() -> BGEM3FlagModel:
    """Carga BGE-M3 en GPU cuando está disponible y, en caso contrario, en CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Cargando el modelo %s en %s.", MODEL_NAME, device)

    return BGEM3FlagModel(
        MODEL_NAME,
        use_fp16=device == "cuda",
        device=device,
    )


def generar_embeddings(
    model: BGEM3FlagModel,
    textos: list[str],
) -> dict[str, Any]:
    """Genera embeddings densos, dispersos y ColBERT para todos los textos."""
    logger.info("Generando embeddings para %s textos.", len(textos))

    embeddings = model.encode(
        textos,
        batch_size=BATCH_SIZE_EMBEDDINGS,
        max_length=MAX_SEQUENCE_LENGTH,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )

    claves_requeridas = {
        "dense_vecs",
        "lexical_weights",
        "colbert_vecs",
    }
    claves_ausentes = claves_requeridas.difference(embeddings)

    if claves_ausentes:
        raise KeyError(
            "El modelo no devolvió todas las representaciones esperadas: "
            f"{sorted(claves_ausentes)}"
        )

    return embeddings


def validar_alineacion_embeddings(
    chunks: list[dict[str, Any]],
    embeddings: dict[str, Any],
) -> None:
    """Comprueba que cada chunk tenga exactamente un embedding de cada tipo."""
    total_chunks = len(chunks)
    dense_vecs = embeddings["dense_vecs"]
    lexical_weights = embeddings["lexical_weights"]
    colbert_vecs = embeddings["colbert_vecs"]

    if total_chunks != dense_vecs.shape[0]:
        raise ValueError(
            f"Desalineación: {total_chunks} chunks frente a "
            f"{dense_vecs.shape[0]} embeddings densos."
        )

    if total_chunks != len(lexical_weights):
        raise ValueError(
            f"Desalineación: {total_chunks} chunks frente a "
            f"{len(lexical_weights)} embeddings dispersos."
        )

    if total_chunks != len(colbert_vecs):
        raise ValueError(
            f"Desalineación: {total_chunks} chunks frente a "
            f"{len(colbert_vecs)} embeddings ColBERT."
        )


def guardar_embeddings(
    chunks: list[dict[str, Any]],
    embeddings: dict[str, Any],
) -> None:
    """Guarda los embeddings y los chunks filtrados respetando el mismo orden."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(DENSE_VECTORS_PATH, embeddings["dense_vecs"])

    with SPARSE_VECTORS_PATH.open("wb") as file:
        pickle.dump(embeddings["lexical_weights"], file)

    with COLBERT_VECTORS_PATH.open("wb") as file:
        pickle.dump(embeddings["colbert_vecs"], file)

    with FILTERED_CHUNKS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Embeddings guardados en %s.", EMBEDDINGS_DIR)
    logger.info("Embeddings densos: %s", embeddings["dense_vecs"].shape)
    logger.info(
        "Embeddings dispersos: %s",
        len(embeddings["lexical_weights"]),
    )
    logger.info(
        "Embeddings ColBERT: %s",
        len(embeddings["colbert_vecs"]),
    )


# ---------------------------------------------------------------------------
# CARGA DE EMBEDDINGS PARA QDRANT
# ---------------------------------------------------------------------------

def cargar_recursos_indexacion() -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    list[dict[int, float]],
]:
    """Carga los chunks filtrados y los embeddings utilizados por Qdrant."""
    rutas_necesarias = (
        FILTERED_CHUNKS_PATH,
        DENSE_VECTORS_PATH,
        SPARSE_VECTORS_PATH,
    )

    for ruta in rutas_necesarias:
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró el recurso necesario para Qdrant: {ruta}"
            )

    with FILTERED_CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    dense_vecs = np.load(DENSE_VECTORS_PATH)

    with SPARSE_VECTORS_PATH.open("rb") as file:
        lexical_weights = pickle.load(file)

    if len(chunks) != dense_vecs.shape[0]:
        raise ValueError(
            f"Desalineación: {len(chunks)} chunks frente a "
            f"{dense_vecs.shape[0]} embeddings densos."
        )

    if len(chunks) != len(lexical_weights):
        raise ValueError(
            f"Desalineación: {len(chunks)} chunks frente a "
            f"{len(lexical_weights)} embeddings dispersos."
        )

    return chunks, dense_vecs, lexical_weights


# ---------------------------------------------------------------------------
# CREACIÓN E INDEXACIÓN DE QDRANT
# ---------------------------------------------------------------------------

def generar_id_punto(indice: int) -> str:
    """
    Genera un UUID determinista para cada posición del conjunto de chunks.

    Al utilizar UUID5, el mismo índice produce siempre el mismo identificador.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"chunk_{indice}"))


def crear_cliente_qdrant(recrear: bool = True) -> QdrantClient:
    """
    Inicializa el cliente local de Qdrant.

    Cuando ``recrear`` es verdadero, elimina completamente la base local
    existente antes de construir la nueva colección.
    """
    if recrear and QDRANT_PATH.exists():
        logger.warning(
            "Se eliminará la base local de Qdrant existente: %s",
            QDRANT_PATH,
        )
        shutil.rmtree(QDRANT_PATH)

    return QdrantClient(path=str(QDRANT_PATH))


def crear_coleccion_qdrant(
    client: QdrantClient,
    dimension_dense: int,
) -> None:
    """Crea la colección con un espacio vectorial denso y otro disperso."""
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=dimension_dense,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(),
            )
        },
    )

    logger.info("Colección %s creada correctamente.", COLLECTION_NAME)


def construir_punto_qdrant(
    indice: int,
    chunk: dict[str, Any],
    vector_dense: np.ndarray,
    vector_sparse: dict[int, float],
) -> PointStruct:
    """Construye un punto de Qdrant con sus vectores y metadatos."""
    payload = dict(chunk)

    # Este índice permite recuperar posteriormente el embedding ColBERT
    # correspondiente desde el archivo colbert_vecs.pkl.
    payload["_indice_chunk"] = indice

    return PointStruct(
        id=generar_id_punto(indice),
        vector={
            "dense": vector_dense.tolist(),
            "sparse": SparseVector(
                indices=[int(clave) for clave in vector_sparse.keys()],
                values=[float(valor) for valor in vector_sparse.values()],
            ),
        },
        payload=payload,
    )


def indexar_chunks_en_qdrant(
    client: QdrantClient,
    chunks: list[dict[str, Any]],
    dense_vecs: np.ndarray,
    lexical_weights: list[dict[int, float]],
) -> None:
    """Inserta todos los chunks en Qdrant por lotes."""
    lote: list[PointStruct] = []

    for indice, chunk in enumerate(chunks):
        punto = construir_punto_qdrant(
            indice=indice,
            chunk=chunk,
            vector_dense=dense_vecs[indice],
            vector_sparse=lexical_weights[indice],
        )
        lote.append(punto)

        if len(lote) >= BATCH_SIZE_QDRANT:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=lote,
            )
            lote.clear()

    if lote:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=lote,
        )

    logger.info(
        "Se han indexado %s chunks en la colección %s.",
        len(chunks),
        COLLECTION_NAME,
    )


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def generar_y_guardar_embeddings() -> None:
    """Ejecuta el proceso completo de generación y exportación de embeddings."""
    chunks_todos = cargar_chunks(CHUNKS_INPUT_PATH)
    chunks = filtrar_chunks_vectorizables(chunks_todos)
    textos = obtener_textos_vectorizacion(chunks)

    model = cargar_modelo_embeddings()
    embeddings = generar_embeddings(model, textos)

    validar_alineacion_embeddings(chunks, embeddings)
    guardar_embeddings(chunks, embeddings)


def construir_qdrant() -> None:
    """Reconstruye la base local de Qdrant a partir de los embeddings guardados."""
    chunks, dense_vecs, lexical_weights = cargar_recursos_indexacion()

    logger.info("Chunks que se indexarán en Qdrant: %s", len(chunks))

    client = crear_cliente_qdrant(recrear=True)
    crear_coleccion_qdrant(
        client=client,
        dimension_dense=dense_vecs.shape[1],
    )
    indexar_chunks_en_qdrant(
        client=client,
        chunks=chunks,
        dense_vecs=dense_vecs,
        lexical_weights=lexical_weights,
    )

    logger.info("Colecciones disponibles: %s", client.get_collections())


def main() -> None:
    """Genera los embeddings y reconstruye Qdrant."""
    generar_y_guardar_embeddings()
    construir_qdrant()
    logger.info("Proceso completado correctamente.")


if __name__ == "__main__":
    main()
