"""
API local para responder correos administrativos mediante un sistema RAG híbrido.

El flujo principal de la aplicación es:
1. Cargar los fragmentos documentales y sus representaciones vectoriales.
2. Recuperar candidatos mediante búsqueda densa y dispersa en Qdrant.
3. Fusionar y reordenar resultados con RRF, ColBERT y un cross-encoder.
4. Generar variantes de la consulta para mejorar la recuperación.
5. Decidir si existe información suficiente para responder.
6. Redactar una respuesta fundamentada y verificar que no contenga invenciones.
7. Exponer el sistema mediante una API FastAPI y una interfaz HTML.

Este archivo contiene únicamente el código necesario para ejecutar la aplicación.
Los bloques de evaluación, diagnóstico y pruebas manuales se han eliminado.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from FlagEmbedding import BGEM3FlagModel
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from scipy.special import expit
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Configuración básica del registro de actividad de la aplicación.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Nombre de la colección vectorial y rutas de los recursos locales.
COLLECTION_NAME = "uv_practicas"
QDRANT_PATH = Path("./qdrant_hibrido")
EMBEDDINGS_DIR = Path("./embeddings_export")
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks_filtrados.json"
DENSE_VECTORS_PATH = EMBEDDINGS_DIR / "dense_vecs.npy"
SPARSE_VECTORS_PATH = EMBEDDINGS_DIR / "lexical_weights.pkl"
COLBERT_VECTORS_PATH = EMBEDDINGS_DIR / "colbert_vecs.pkl"
TRAIN_EXAMPLES_PATH = Path("./train_correos.json")
INTERFACE_PATH = Path("./rag_interfaz.html")

# ---------------------------------------------------------------------------
# CARGA DE LOS RECURSOS PRECALCULADOS
# ---------------------------------------------------------------------------

# Los chunks y embeddings se generan previamente y deben conservar el mismo orden.
with CHUNKS_PATH.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

dense_vecs = np.load(DENSE_VECTORS_PATH)

with SPARSE_VECTORS_PATH.open("rb") as file:
    lexical_weights = pickle.load(file)

with COLBERT_VECTORS_PATH.open("rb") as file:
    colbert_vecs = pickle.load(file)

assert len(chunks) == dense_vecs.shape[0], (
    f"Desalineación: {len(chunks)} chunks vs {dense_vecs.shape[0]} dense embeddings."
)

assert len(chunks) == len(lexical_weights), (
    f"Desalineación: {len(chunks)} chunks vs {len(lexical_weights)} sparse embeddings."
)

assert len(chunks) == len(colbert_vecs), (
    f"Desalineación: {len(chunks)} chunks vs {len(colbert_vecs)} colbert embeddings."
)

# ---------------------------------------------------------------------------
# INICIALIZACIÓN DE MODELOS Y BASE VECTORIAL
# ---------------------------------------------------------------------------

# Se conserva esta variable para registrar si hay GPU disponible.
device = "cuda" if torch.cuda.is_available() else "cpu"

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=False,  # Para evitar problemas de memoria en GPU
    device="cpu"
)

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    device="cpu",
    max_length=1024
)

client = QdrantClient(path=str(QDRANT_PATH))

logger.info("Modelos y recursos cargados correctamente.")
logger.info("Dispositivo disponible: %s", device)
logger.info("Chunks disponibles: %s", len(chunks))
logger.info("Forma de los embeddings densos: %s", dense_vecs.shape)
logger.info("Embeddings dispersos disponibles: %s", len(lexical_weights))
logger.info("Embeddings ColBERT disponibles: %s", len(colbert_vecs))

# ---------------------------------------------------------------------------
# RECUPERACIÓN HÍBRIDA DE DOCUMENTOS
# ---------------------------------------------------------------------------

def buscar_dense_sparse(pregunta, top_k=30):
    query_embedding = model.encode(
        [pregunta], max_length=8192,
        return_dense=True, return_sparse=True, return_colbert_vecs=True
    )
    dense_query = query_embedding["dense_vecs"][0].tolist()
    sparse_query = query_embedding["lexical_weights"][0]
    colbert_query = query_embedding["colbert_vecs"][0]

    dense = client.query_points(
        collection_name=COLLECTION_NAME, query=dense_query, using="dense", limit=top_k
    ).points
    sparse = client.query_points(
        collection_name=COLLECTION_NAME,
        query=SparseVector(
            indices=[int(k) for k in sparse_query.keys()],
            values=[float(v) for v in sparse_query.values()]
        ),
        using="sparse", limit=top_k
    ).points

    return dense, sparse, colbert_query


def fusionar_rrf_dense_sparse(dense, sparse, top_k=30):
    scores = {}
    objetos = {}
    k = 60
    for resultados in [dense, sparse]:
        for rank, r in enumerate(resultados):
            scores[r.id] = scores.get(r.id, 0) + 1 / (k + rank + 1)
            objetos[r.id] = r
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [objetos[id_] for id_, score in ranking[:top_k]]


def score_colbert(query_colbert, doc_colbert):
    q = np.asarray(query_colbert, dtype=np.float32)
    d = np.asarray(doc_colbert, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    d_norm = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    sims = q_norm @ d_norm.T
    return float(sims.max(axis=1).sum())


def rerank_colbert_solo_candidatos(candidatos, query_colbert, top_k=20):
    puntuados = []
    for r in candidatos:
        idx = r.payload["_indice_chunk"]
        doc_colbert = colbert_vecs[idx]
        score = score_colbert(query_colbert, doc_colbert)
        puntuados.append((r, score))
    puntuados = sorted(puntuados, key=lambda x: x[1], reverse=True)
    return puntuados[:top_k]


def recuperar_con_reranker(pregunta, top_k_busqueda=30, top_k_colbert=20, top_k_final=10):
    dense, sparse, query_colbert = buscar_dense_sparse(pregunta, top_k=top_k_busqueda)
    candidatos = fusionar_rrf_dense_sparse(dense, sparse, top_k=top_k_busqueda)
    candidatos_colbert = rerank_colbert_solo_candidatos(
        candidatos=candidatos, query_colbert=query_colbert, top_k=top_k_colbert
    )
    pares = [
        (pregunta, r.payload.get("texto_vector", r.payload.get("contenido", "")))
        for r, score_colbert_val in candidatos_colbert
    ]
    scores_reranker = reranker.predict(pares)
    rerankeados = sorted(zip(candidatos_colbert, scores_reranker), key=lambda x: x[1], reverse=True)
    return rerankeados[:top_k_final]


# ---------------------------------------------------------------------------
# MODELO DE LENGUAJE
# ---------------------------------------------------------------------------

# Modelo generativo cuantizado a 4 bits para reducir el consumo de memoria.
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# 2. Definimos la configuración de cuantización a 4 bits
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer_llm = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3. Cargamos el modelo aplicando la cuantización
model_llm = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model_llm.eval()

# Reformula el correo como una consulta administrativa adecuada para el retrieval.
def reescribir_query(correo, max_new_tokens=150):
    system_prompt = (
        "Eres un asistente que reformula consultas de correos electrónicos en preguntas "
        "formales usando terminología legal y administrativa española exacta, tal y como "
        "aparecería en un Boletín Oficial del Estado. Sustituye términos coloquiales por "
        "su equivalente técnico/legal cuando exista (por ejemplo, 'becario' debe sustituirse "
        "por el término legal correspondiente si lo conoces). "
        "El correo puede estar en castellano o en valenciano — en ambos casos reformula "
        "la pregunta en castellano formal. "
        "Responde ÚNICAMENTE con la pregunta reformulada, sin explicaciones ni texto adicional."
    )
    mensajes = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Correo: {correo}\n\nPregunta reformulada:"}
    ]
    
    texto_input = tokenizer_llm.apply_chat_template(
        mensajes, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer_llm(texto_input, return_tensors="pt").to(model_llm.device)
    
    with torch.no_grad():
        output = model_llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer_llm.eos_token_id
        )
        
    respuesta = tokenizer_llm.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return respuesta.strip()

# ---------------------------------------------------------------------------
# AMPLIACIÓN DEL CONTEXTO DOCUMENTAL
# ---------------------------------------------------------------------------

# Índice auxiliar para localizar rápidamente cualquier chunk por su identificador.
chunks_por_id = {c["metadatos"]["chunk_id"]: c for c in chunks}

def obtener_hermanos(chunk_id_actual, parent_id, chunks_por_id):
    if parent_id is None:
        return []
    try:
        num_actual = int(chunk_id_actual.split(".")[-1])
    except (ValueError, IndexError):
        return []
    ids_hermanos = [f"{parent_id}.{num_actual - 1}", f"{parent_id}.{num_actual + 1}"]
    return [chunks_por_id[hid] for hid in ids_hermanos if hid in chunks_por_id]

# Construye el contexto final incluyendo los fragmentos vecinos del mismo padre.
def preparar_contexto_con_hermanos(resultados_reranker, chunks_por_id, max_chunks=5):
    incluidos = {}
    orden = []
    for (r, _colbert_score_val), _reranker_score_val in resultados_reranker[:max_chunks]:
        metadatos = r.payload.get("metadatos", {})
        chunk_id = metadatos.get("chunk_id")
        parent_id = metadatos.get("parent_chunk")
        if chunk_id not in incluidos:
            incluidos[chunk_id] = {"contenido": r.payload.get("contenido", ""), "metadatos": metadatos}
            orden.append(chunk_id)
        for h in obtener_hermanos(chunk_id, parent_id, chunks_por_id):
            h_id = h["metadatos"]["chunk_id"]
            if h_id not in incluidos:
                incluidos[h_id] = {"contenido": h["contenido"], "metadatos": h["metadatos"]}
                orden.append(h_id)

    contexto = ""
    fuentes = set()
    for i, cid in enumerate(orden, start=1):
        c = incluidos[cid]
        archivo = c["metadatos"].get("Archivo_Origen", "desconocido")
        contexto += f"[FRAGMENTO {i}] (chunk_id={cid}, Fuente: {archivo})\n{c['contenido']}\n\n"
        fuentes.add(archivo)

    
    return contexto, list(fuentes)


# ---------------------------------------------------------------------------
# EJEMPLOS FEW-SHOT PARA EL ESTILO DE REDACCIÓN
# ---------------------------------------------------------------------------
def indexar_dataset_ejemplos(ruta_json: str | Path):
    """Carga los ejemplos históricos y calcula sus embeddings densos."""
    ruta_json = Path(ruta_json)
    with ruta_json.open("r", encoding="utf-8") as file:
        datos = json.load(file)
    correos = [d["correo_consulta"] for d in datos]
    embeddings_dataset = model.encode(correos, max_length=8192, return_dense=True)["dense_vecs"]
    return datos, embeddings_dataset


def encontrar_ejemplos_similares(correo_original, datos, embeddings_dataset, n=2):
    query_emb = model.encode([correo_original], max_length=8192, return_dense=True)["dense_vecs"][0]
    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    d = embeddings_dataset / (np.linalg.norm(embeddings_dataset, axis=1, keepdims=True) + 1e-9)
    sims = d @ q
    top_indices = np.argsort(sims)[::-1][:n]
    return [datos[i] for i in top_indices]


# ---------------------------------------------------------------------------
# PROMPTS Y FUNCIONES DEL PIPELINE DE RESPUESTA
# ---------------------------------------------------------------------------

MENSAJE_DERIVAR = (
    "Gracias por tu consulta. En este momento no dispongo de información suficiente "
    "para responderte con seguridad, por lo que te recomiendo contactar directamente "
    "con la secretaría, que podrá ayudarte con este asunto."
)

GUION_CHAIN_OF_THOUGHT = """
PASO 0 - COMPROBACIÓN PREVIA (preguntas sobre el caso personal del remitente):
Antes de nada, comprueba si el correo pregunta por el ESTADO PERSONAL o el CASO
CONCRETO del remitente, en vez de por información general del procedimiento.
Ejemplos de este tipo de preguntas:
- "¿Habéis recibido mi documentación / mi justificante / mi solicitud?"
- "¿Ya habéis revisado / evaluado mi expediente?"
- "¿Cuándo me confirmaréis mi plaza / mi matrícula?"
- "¿Está ya tramitada mi beca / mi certificado?"
- "¿Por qué no he recibido todavía la respuesta que me prometisteis?"

Estas preguntas dependen de datos internos sobre ESA persona concreta (si su email
llegó, si su expediente ya se revisó, etc.), que NUNCA están en la documentación
general del RAG, aunque el buscador encuentre fragmentos que hablan del
procedimiento general (p.ej. "cómo se envía la documentación" o "plazos de
revisión"). Encontrar ese tipo de fragmento NO significa que se pueda confirmar
el caso personal del remitente.

Si el correo es de este tipo -> DECISION: DERIVAR, directamente, sin pasar al
PASO 1. No importa el score de los fragmentos ni si tratan el tema en general.

Si el correo pregunta por el PROCEDIMIENTO o REQUISITOS en general (no por su
caso personal ya iniciado) -> continúa al PASO 1 normalmente. Por ejemplo,
"¿cómo debo enviar mi documentación?" sí es una pregunta de procedimiento general
(PASO 1), mientras que "¿ya habéis recibido la que os envié ayer?" es sobre su
caso personal (PASO 0 -> DERIVAR).

PASO 1 - CRITERIO GENERAL:
Para el resto de preguntas, hazte una única pregunta: ¿los fragmentos
proporcionados tratan el tema concreto por el que se pregunta en el correo, y
contienen algún dato, requisito o regla que permita construir una respuesta?

- Si SÍ (aunque tengas que aplicar ese dato o regla al caso concreto descrito en
  el correo, o aunque no cubra absolutamente todos los matices posibles) ->
  DECISION: RESPONDER

- Si NO (los fragmentos no mencionan en absoluto ese tema, o tratan un tema
  distinto que solo coincide superficialmente) -> DECISION: DERIVAR

Por defecto, RESPONDE. Deriva únicamente cuando estés seguro de que los
fragmentos no contienen ninguna información relacionada con lo que se pregunta.
No derives solo porque la respuesta requiera leer con atención, combinar dos
frases del mismo fragmento, o aplicar una regla general al caso particular del
correo — eso sigue siendo DECISION: RESPONDER.
"""

SYSTEM_PROMPT_MULTIQUERY = """
Eres un asistente que genera variantes de búsqueda para un sistema RAG universitario.
Dado un correo de un usuario, genera 3 formulaciones ALTERNATIVAS de la pregunta que
hay que buscar en la base de documentación, cada una en una línea, sin numerar ni
añadir texto adicional. Las variantes deben:
- Usar sinónimos y términos administrativos alternativos (p.ej. "cuota" / "afiliación" / "cotización").
- Incluir al menos una variante muy literal, casi copiando la frase clave del correo.
- Incluir al menos una variante más general por si la pregunta usa un término muy específico
  que no coincide con el vocabulario de la normativa.
No respondas la pregunta, solo genera las variantes de búsqueda.
"""

SYSTEM_PROMPT_DECISION = (
    """
    Eres un asistente administrativo de la Universitat de València especializado
    en prácticas universitarias y títulos propios. Tu ÚNICA tarea ahora es decidir
    si, con los fragmentos de documentación proporcionados, se puede responder con
    certeza a la pregunta concreta del correo recibido.

    NO redactes ninguna respuesta al correo. NO escribas nada de texto adicional.
    Tu salida debe ser EXCLUSIVAMENTE una de estas dos líneas, sin nada más:

    DECISION: RESPONDER
    DECISION: DERIVAR
    """
    + GUION_CHAIN_OF_THOUGHT
)

SYSTEM_PROMPT_REDACCION = """
Eres un asistente administrativo de la Universitat de València especializado
en prácticas universitarias y títulos propios. Ya se ha determinado que los
fragmentos de documentación proporcionados SÍ contienen información suficiente
para responder con certeza a este correo. Tu tarea ahora es únicamente redactar
esa respuesta.

Reglas estrictas:
- No inventes ni asumas ningún dato, fecha, nombre, cifra o normativa que no esté
  literalmente en los fragmentos proporcionados.
- Cada afirmación relevante de tu respuesta (plazos, cifras, requisitos, obligaciones)
  debe poder localizarse literalmente en los fragmentos. Si no puedes localizarla,
  no la incluyas.
- Usa un tono formal y profesional, similar al de los ejemplos proporcionados.
- Los ejemplos que se te dan a continuación son SOLO para que imites el estilo,
  el tono y el formato de redacción. No implican nada sobre si este correo en
  concreto debe recibir respuesta: esa decisión ya ha sido tomada y es "sí".
- Los fragmentos pueden estar en castellano o en valenciano; responde siempre
  en castellano.
- No incluyas ninguna etiqueta de decisión, solo el texto de la respuesta al correo.

- NO repitas, resumas ni parafrasees la consulta del remitente a modo de
  introducción (evita frases como "en respuesta a su consulta...", "entendemos
  que están interesados en...", "gracias por contactarnos sobre..."). El
  remitente ya sabe lo que ha preguntado. Ve directo a la información relevante
  desde la primera frase.
- No añadas al final ninguna referencia a los fragmentos, documentos o fuentes
  utilizadas (por ejemplo, no escribas "Fragmentos utilizados: ..." ni nada
  similar). Esa información se gestiona aparte, fuera del texto de la respuesta.
- Termina SIEMPRE la respuesta con el cierre exacto:
  "Contenido generado por IA"
  No inventes ni cambies este cierre por ningún nombre propio, cargo o firma
  distinta, bajo ninguna circunstancia.

- Si la pregunta requiere calcular si una fecha, año, plazo o periodo sigue vigente
  o ya ha terminado, NO hagas el cálculo.
  Limítate a indicar literalmente el plazo o requisito que aparece en los fragmentos.
  Por ejemplo: "la normativa establece un plazo de tres años desde la finalización
  de los estudios".
  No digas "todavía estás dentro del plazo", "ya no estás dentro del plazo",
  "puedes hacerlo" o "no puedes hacerlo", salvo que esa conclusión aparezca
  literalmente en los fragmentos.
"""

SYSTEM_PROMPT_VERIFICACION = """
Eres un verificador de fundamentación (fact-checker) para un asistente administrativo
universitario. Se te da un correo, unos fragmentos de documentación oficial, y un
borrador de respuesta ya redactado. Tu única tarea es comprobar si el borrador
INVENTA datos concretos que NO están en los fragmentos.

Cuenta como NO SOPORTADO (FALLO) únicamente:
- Cifras, importes, porcentajes o plazos que no aparecen en los fragmentos.
- Fechas, nombres propios, artículos de ley o normativas que no aparecen en los fragmentos.
- Afirmaciones categóricas ("siempre", "nunca", "en todos los casos") que los
  fragmentos no respaldan.

Cuenta también como NO SOPORTADO (FALLO):
- Cualquier confirmación sobre el estado personal o el caso concreto del
  remitente (p.ej. "hemos recibido tu documentación", "tu expediente ya está
  revisado", "tu plaza está confirmada") si esa confirmación no puede venir de
  documentación general, sino de datos internos sobre esa persona en concreto.

NO cuenta como fallo, y debes marcar OK, cuando el borrador:
- Aplica una regla o requisito general del fragmento al caso concreto del correo.
- Parafrasea o reformula el contenido de los fragmentos sin añadir datos nuevos.
- Combina información de varios fragmentos proporcionados sin inventar nada.
- Da una respuesta razonada (p.ej. una negación o conclusión lógica directa) a
  partir de un dato explícito en los fragmentos.

Ante la duda, marca OK. Solo marca FALLO si hay una invención clara y concreta.

Responde ÚNICAMENTE con una de estas dos líneas, sin nada más:

VERIFICACION: OK
VERIFICACION: FALLO
"""


def _extraer_etiqueta(texto, etiqueta, valores_validos, valor_por_defecto):
    """Parsea 'ETIQUETA: VALOR' de forma robusta. Fail-safe: valor_por_defecto si no se encuentra."""
    patron = rf"{etiqueta}:\s*({'|'.join(valores_validos)})"
    match = re.search(patron, texto, re.IGNORECASE)
    if not match:
        return valor_por_defecto
    return match.group(1).upper()


def _generar(mensajes, temperature, max_new_tokens, do_sample):
    texto_input = tokenizer_llm.apply_chat_template(
        mensajes, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer_llm(texto_input, return_tensors="pt").to(model_llm.device)

    with torch.no_grad():
        output = model_llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer_llm.eos_token_id,
        )

    return tokenizer_llm.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def _generar_variantes_query(correo_original, pregunta_base):
    """Genera variantes de búsqueda además de la pregunta reescrita original,
    para mejorar el recall cuando la reescritura 'base' no encuentra el chunk correcto."""
    mensajes = [
        {"role": "system", "content": SYSTEM_PROMPT_MULTIQUERY},
        {"role": "user", "content": f"CORREO:\n{correo_original}"},
    ]
    texto = _generar(mensajes, temperature=0.4, max_new_tokens=200, do_sample=True)
    variantes = [l.strip("-• \t") for l in texto.split("\n") if l.strip()]
    variantes = [v for v in variantes if len(v) > 5][:3]

    todas = [pregunta_base] + variantes
    # dedup manteniendo orden
    vistas = set()
    unicas = []
    for q in todas:
        if q.lower() not in vistas:
            vistas.add(q.lower())
            unicas.append(q)
    return unicas


def _obtener_id_chunk(item):
    """Extrae el id real del ScoredPoint anidado en un resultado de
    recuperar_con_reranker. La forma observada es:
        item = ((ScoredPoint(id=..., ...), score_intermedio), score_final)
    así que el id está en item[0][0].id. Se dejan fallbacks por si la forma
    cambia en algún caso (por ejemplo, si algún día cambia el reranker)."""
    try:
        return item[0][0].id
    except (IndexError, AttributeError, TypeError):
        pass

    # Fallback: quizá el ScoredPoint viene en primer nivel, no anidado
    try:
        if hasattr(item[0], "id"):
            return item[0].id
    except (IndexError, TypeError):
        pass

    # Último recurso: nunca debe lanzar TypeError, aunque no dedupe bien
    return f"objid::{id(item)}"


def _recuperar_multi_query(queries, top_k_busqueda, top_k_colbert, top_k_final):
    """Ejecuta recuperar_con_reranker con cada query y fusiona resultados por el
    id real del chunk (ScoredPoint.id), quedándose con el score final máximo
    (item[1], el mismo que luego se normaliza con expit). Devuelve los items en
    su formato original íntegro, para que preparar_contexto_con_hermanos y el
    resto del pipeline los consuman exactamente igual que antes."""
    mejores_por_id = {}  # id -> (item_completo, score_final)

    for q in queries:
        resultados_q = recuperar_con_reranker(
            pregunta=q,
            top_k_busqueda=top_k_busqueda,
            top_k_colbert=top_k_colbert,
            top_k_final=top_k_final,
        )
        for item in resultados_q:
            chunk_id = _obtener_id_chunk(item)
            score_final = float(item[1])
            if chunk_id not in mejores_por_id or score_final > mejores_por_id[chunk_id][1]:
                mejores_por_id[chunk_id] = (item, score_final)

    fusionados = sorted(mejores_por_id.values(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in fusionados[:top_k_final]]


def responder_correo_completo(
    correo_original,
    datos_ejemplos,
    embeddings_ejemplos,
    chunks_por_id,
    top_k_busqueda=100,
    top_k_colbert=50,
    top_k_final=3,
    umbral_reranker=0.45,
):
    pregunta_retrieval = reescribir_query(correo_original)

    # ---------- RECUPERACIÓN MULTI-QUERY (mejora recall) ----------
    queries = _generar_variantes_query(correo_original, pregunta_retrieval)
    resultados = _recuperar_multi_query(
        queries, top_k_busqueda=top_k_busqueda, top_k_colbert=top_k_colbert, top_k_final=top_k_final
    )

    if not resultados:
        return {
            "respuesta": MENSAJE_DERIVAR,
            "fuentes": [],
            "score_maximo": None,
            "decision": "DERIVAR",
            "motivo": "sin_resultados_recuperacion",
            "queries_usadas": queries,
        }

    mejor_score_raw = float(resultados[0][1])
    mejor_score_norm = float(expit(mejor_score_raw))

    if mejor_score_norm < umbral_reranker:
        return {
            "respuesta": MENSAJE_DERIVAR,
            "fuentes": [],
            "score_maximo": mejor_score_norm,
            "decision": "DERIVAR",
            "motivo": "umbral_reranker",
            "queries_usadas": queries,
        }

    contexto, fuentes = preparar_contexto_con_hermanos(
        resultados, chunks_por_id, max_chunks=top_k_final
    )

    # ---------- LLAMADA 1: SOLO DECISIÓN (sin few-shot) ----------
    prompt_decision = (
        f"CORREO RECIBIDO:\n{correo_original}\n\n"
        f"FRAGMENTOS DE DOCUMENTACIÓN DISPONIBLES:\n{contexto}\n\n"
        f"Decide si se puede responder con certeza. Responde ÚNICAMENTE con la línea DECISION."
    )
    mensajes_decision = [
        {"role": "system", "content": SYSTEM_PROMPT_DECISION},
        {"role": "user", "content": prompt_decision},
    ]
    texto_decision = _generar(mensajes_decision, temperature=0.0, max_new_tokens=300, do_sample=False)
    decision = _extraer_etiqueta(texto_decision, "DECISION", ["RESPONDER", "DERIVAR"], "DERIVAR")

    if decision == "DERIVAR":
        return {
            "respuesta": MENSAJE_DERIVAR,
            "fuentes": [],
            "score_maximo": mejor_score_norm,
            "decision": "DERIVAR",
            "motivo": "decision_llm",
            "queries_usadas": queries,
            "texto_decision_raw": texto_decision,
        }

    # ---------- LLAMADA 2: SOLO REDACCIÓN (aquí sí entra el few-shot, solo para estilo) ----------
    ejemplos = encontrar_ejemplos_similares(correo_original, datos_ejemplos, embeddings_ejemplos, n=2)
    ejemplos_texto = ""
    for i, ej in enumerate(ejemplos, start=1):
        ejemplos_texto += (
            f"EJEMPLO {i}:\nCorreo: {ej['correo_consulta']}\nRespuesta: {ej['correo_respuesta']}\n\n"
        )

    prompt_redaccion = (
        f"CORREO RECIBIDO:\n{correo_original}\n\n"
        f"FRAGMENTOS DE DOCUMENTACIÓN DISPONIBLES:\n{contexto}\n\n"
        f"---\n\n"
        f"A continuación tienes ejemplos de correos previos y cómo se respondieron, "
        f"únicamente como referencia de estilo y tono:\n\n{ejemplos_texto}"
        f"---\n\n"
        f"Redacta ahora la respuesta al correo recibido, basándote solo en los fragmentos.\n"
        f"Si hay fechas o plazos, NO calcules si están vigentes o vencidos. "
        f"Indica únicamente el plazo literal que aparece en la documentación."
    )

    mensajes_redaccion = [
        {"role": "system", "content": SYSTEM_PROMPT_REDACCION},
        {"role": "user", "content": prompt_redaccion},
    ]
    respuesta = _generar(mensajes_redaccion, temperature=0.0, max_new_tokens=1200, do_sample=False)

    # ---------- LLAMADA 3: VERIFICACIÓN DE GROUNDING (solo aviso, ya NO bloquea la respuesta) ----------
    prompt_verificacion = (
        f"CORREO:\n{correo_original}\n\n"
        f"FRAGMENTOS:\n{contexto}\n\n"
        f"BORRADOR DE RESPUESTA A VERIFICAR:\n{respuesta}\n\n"
        f"Verifica el borrador según las reglas indicadas."
    )
    mensajes_verificacion = [
        {"role": "system", "content": SYSTEM_PROMPT_VERIFICACION},
        {"role": "user", "content": prompt_verificacion},
    ]
    texto_verificacion = _generar(
        mensajes_verificacion, temperature=0.0, max_new_tokens=100, do_sample=False
    )
    verificacion = _extraer_etiqueta(texto_verificacion, "VERIFICACION", ["OK", "FALLO"], "OK")

    # Ya no se deriva por esto: se registra como aviso para poder revisarlo y
    # ajustar prompts con datos reales, pero la respuesta se devuelve igualmente.
    if verificacion == "FALLO":
        logger.warning(
            "El verificador detectó un posible dato no soportado: %r",
            texto_verificacion,
        )

    return {
        "respuesta": respuesta,
        "fuentes": fuentes,
        "score_maximo": mejor_score_norm,
        "decision": "RESPONDER",
        "motivo": "ok",
        "queries_usadas": queries,
        "aviso_verificacion": verificacion,  # "OK" o "FALLO" (informativo, no bloquea)
        "texto_verificacion_raw": texto_verificacion,
    }


# ---------------------------------------------------------------------------
# API FASTAPI
# ---------------------------------------------------------------------------

# Los ejemplos se indexan una sola vez durante el arranque de la aplicación.
datos_ejemplos, embeddings_ejemplos = indexar_dataset_ejemplos(TRAIN_EXAMPLES_PATH)

app = FastAPI(
    title="Asistente RAG de correos",
    description="API local para generar respuestas administrativas fundamentadas.",
    version="1.0.0",
)

# CORS abierto para facilitar el uso de la interfaz local.
# En un despliegue público conviene sustituir "*" por los dominios autorizados.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Consulta(BaseModel):
    """Datos que debe enviar el cliente al endpoint de generación."""

    correo: str = Field(
        ...,
        min_length=1,
        description="Contenido completo del correo que debe analizarse.",
    )


@app.post("/generar-respuesta")
def generar_respuesta(consulta: Consulta) -> dict[str, Any]:
    """Procesa un correo y devuelve la decisión, la respuesta y sus metadatos."""
    resultado = responder_correo_completo(
        correo_original=consulta.correo,
        datos_ejemplos=datos_ejemplos,
        embeddings_ejemplos=embeddings_ejemplos,
        chunks_por_id=chunks_por_id,
    )

    return {
        "decision": resultado["decision"],
        "respuesta": resultado["respuesta"],
        "fuentes": resultado.get("fuentes", []),
        "score_maximo": resultado.get("score_maximo"),
        "motivo": resultado.get("motivo"),
        "aviso_verificacion": resultado.get("aviso_verificacion"),
    }


@app.get("/")
def interfaz():
    return FileResponse("rag_interfaz.html")
