"""
Pipeline de limpieza y chunking para documentación de postgrado.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------------------

RAW_MARKDOWN_DIR = Path("./datos_postgrado/estructura_igual")
CLEAN_MARKDOWN_DIR = Path("./postgrado_igual_md_limpios")
OUTPUT_CHUNKS_FILE = Path("./chunks_postgrado_iguales.json")

DOMAIN = "postgrado"
FIRST_PARENT_ID = 89


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Títulos principales reconocidos en los documentos originales.
SECTION_TITLES = {
    "Datos generales",
    "Salida profesional",
    "Dirección",
    "Más información",
    "Admisión y matrícula",
    "Documentación a adjuntar",
    "Normas generales",
    "Programa",
    "Profesorado",
    "Metodología",
    "Acceso y Resultados de Aprendizaje",
}


# Campos que deben transformarse en elementos de lista dentro de “Datos generales”.
GENERAL_DATA_FIELDS = [
    "Objetivos",
    "Curso académico",
    "Tipo de curso",
    "Modalidad",
    "Precio matrícula",
    "Fecha fin preinscripción",
    "Fecha de inicio curso",
    "Fecha finalización del curso",
    "Edición",
    "Código título",
    "Créditos",
    "Horario",
    "Lugar de impartición",
    "Teléfono",
    "E-mail",
    "Fin preinscripción",
]


# Campos que pueden aparecer separados de su contenido en la sección de acceso.
ACCESS_FIELDS = [
    "Requisitos titulación",
    "Criterios admisión",
    "Resultados de aprendizaje",
]

# ---------------------------------------------------------------------------
# FUNCIONES AUXILIARES DE LIMPIEZA
# ---------------------------------------------------------------------------

def remove_bold(text: str) -> str:
    """Elimina el marcado Markdown de negrita conservando su contenido."""
    return re.sub(r"\*\*(.*?)\*\*", r"\1", text)


def strip_markdown_heading(line: str) -> str:
    """Elimina los símbolos de encabezado Markdown de una línea."""
    return re.sub(r"^#{1,6}\s*", "", line).strip()


def is_markdown_table_row(line: str) -> bool:
    """Indica si una línea pertenece a una tabla Markdown."""
    return line.strip().startswith("|")


def normalize_bullet(line: str) -> str:
    """
    Normaliza distintos símbolos de viñeta al formato '- '.

    Las enumeraciones numéricas y alfabéticas se conservan para no alterar
    la estructura original del contenido.
    """
    line = line.strip()

    if re.match(r"^\d+(\.\d+)*[\.)-]?\s+", line):
        return line

    if re.match(r"^[a-zA-Z][\.)]\s+", line):
        return line

    return re.sub(r"^[*+•●▪■□◦‣⁃∙]\s+", "- ", line)


def split_joined_enumerations(text: str) -> str:
    """Separa enumeraciones que aparecen pegadas al texto anterior."""
    text = re.sub(r"(\d+(?:\.\d+)*)\.-", r"\1.", text)

    text = re.sub(
        r"(?<!\n)\s+(\d+(?:\.\d+)*\.\s+(?=[A-ZÁÉÍÓÚÀÈÍÒÚÜÑÇ]))",
        r"\n\1",
        text,
    )

    # Algunos programas contienen bloques como “Mòdul 1.” pegados al texto.
    return re.sub(r"(?<!\n)(M[oòó]dul\s+\d+\.)", r"\n\1", text)


def normalize_joined_general_field(line: str) -> str:
    """
    Separa un campo de datos generales de su valor cuando aparecen pegados.

    Ejemplo:
        'ModalidadOnline' -> '- Modalidad: Online'
    """
    for field in GENERAL_DATA_FIELDS:
        if line.startswith(field) and not line.startswith(f"{field}:"):
            value = line[len(field):].strip()
            if value:
                return f"- {field}: {value}"

    return line


def convert_markdown_table(
    lines: list[str],
    start_index: int,
) -> tuple[list[str], int]:
    """
    Convierte una tabla de profesorado en una lista legible.

    Si la tabla no tiene el formato esperado, sus filas se conservan para
    evitar una pérdida accidental de información.
    """
    table_rows: list[str] = []
    index = start_index

    while index < len(lines) and is_markdown_table_row(lines[index]):
        table_rows.append(lines[index].strip())
        index += 1

    if len(table_rows) < 3:
        return table_rows, index

    headers = [
        header.strip()
        for header in table_rows[0].strip("|").split("|")
    ]
    converted_rows: list[str] = []

    for row in table_rows[2:]:
        values = [
            value.strip()
            for value in row.strip("|").split("|")
        ]

        if len(values) != len(headers):
            converted_rows.append(row)
            continue

        data = dict(zip(headers, values))

        first_name = data.get("Nombre", "").strip()
        last_name = data.get("Apellidos", "").strip()
        affiliation = data.get("Vinculación", "").strip()
        additional_info = data.get("+ info", "").strip()

        person = " ".join(
            value for value in (first_name, last_name) if value
        ).strip()

        if not person and not affiliation:
            continue

        if affiliation and additional_info:
            converted_rows.append(
                f"- {person}: {affiliation}. {additional_info}"
            )
        elif affiliation:
            converted_rows.append(f"- {person}: {affiliation}")
        else:
            converted_rows.append(f"- {person}")

    return converted_rows, index


def join_access_fields(lines: list[str]) -> list[str]:
    """
    Une los encabezados de acceso con el texto que aparece a continuación.

    El resultado adopta el formato:
        - Nombre del campo: contenido
    """
    result: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index].strip()

        if line in ACCESS_FIELDS:
            field = line
            content: list[str] = []
            index += 1

            while index < len(lines):
                next_line = lines[index].strip()

                if (
                    next_line in ACCESS_FIELDS
                    or next_line.startswith("##")
                    or next_line.startswith("###")
                ):
                    break

                if next_line:
                    content.append(next_line)

                index += 1

            result.append(f"- {field}: {' '.join(content).strip()}")
            continue

        result.append(lines[index])
        index += 1

    return result


def clean_markdown(text: str) -> str:
    """Aplica todas las reglas de limpieza a un documento Markdown."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = remove_bold(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = split_joined_enumerations(text)

    original_lines = text.split("\n")
    cleaned_lines: list[str] = []

    is_first_title = True
    previous_title: str | None = None
    inside_access_section = False

    index = 0

    while index < len(original_lines):
        line = original_lines[index].strip()

        if not line:
            index += 1
            continue

        if is_markdown_table_row(line):
            converted_table, next_index = convert_markdown_table(
                original_lines,
                index,
            )
            cleaned_lines.extend(converted_table)
            index = next_index
            continue

        # Elimina encabezados basura formados por una sola letra.
        if re.match(r"^#{1,6}\s+[a-zA-Z]$", line):
            index += 1
            continue

        plain_title = strip_markdown_heading(line)

        # La primera línea útil se considera el título del documento.
        if is_first_title:
            cleaned_lines.append(f"## {plain_title}")
            is_first_title = False
            previous_title = plain_title
            index += 1
            continue

        if plain_title in SECTION_TITLES:
            if plain_title != previous_title:
                cleaned_lines.extend(["", f"### {plain_title}"])
                previous_title = plain_title

            inside_access_section = (
                plain_title == "Acceso y Resultados de Aprendizaje"
            )
            index += 1
            continue

        # Evita títulos repetidos dentro de su propia sección.
        if previous_title in {"Dirección", "Metodología"} and plain_title == previous_title:
            index += 1
            continue

        if inside_access_section and plain_title in ACCESS_FIELDS:
            cleaned_lines.append(plain_title)
            index += 1
            continue

        normalized_line = normalize_bullet(line)

        if previous_title == "Datos generales":
            normalized_line = normalize_joined_general_field(normalized_line)

            if (
                not normalized_line.startswith("- ")
                and any(
                    normalized_line.startswith(field)
                    for field in GENERAL_DATA_FIELDS
                )
            ):
                normalized_line = f"- {normalized_line}"

        cleaned_lines.append(normalized_line)
        index += 1

    cleaned_lines = join_access_fields(cleaned_lines)
    cleaned_text = "\n".join(cleaned_lines)

    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = re.sub(r" +", " ", cleaned_text)
    cleaned_text = cleaned_text.replace(
        "### Documentación a adjuntar\nMáster",
        "### Documentación a adjuntar\n\nMáster",
    )

    # Elimina viñetas vacías que hayan podido generarse durante la limpieza.
    cleaned_text = re.sub(r"\n-\s*\n", "\n", cleaned_text)
    cleaned_text = re.sub(r"\n-\s*$", "\n", cleaned_text)

    return cleaned_text.strip() + "\n"


def clean_markdown_directory(
    input_dir: Path,
    output_dir: Path,
) -> int:
    """Limpia todos los archivos Markdown de una carpeta."""
    if not input_dir.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta de entrada: {input_dir}"
        )

    markdown_files = sorted(input_dir.glob("*.md"))

    if not markdown_files:
        raise FileNotFoundError(
            f"No se encontraron archivos .md en {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for source_path in markdown_files:
        cleaned_text = clean_markdown(
            source_path.read_text(encoding="utf-8")
        )
        destination_path = output_dir / source_path.name
        destination_path.write_text(cleaned_text, encoding="utf-8")

        logger.info("Documento limpiado: %s", source_path.name)

    logger.info(
        "Limpieza completada: %s documentos.",
        len(markdown_files),
    )
    return len(markdown_files)

# ---------------------------------------------------------------------------
# FUNCIONES DE CHUNKING
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Elimina espacios verticales excesivos sin modificar el contenido."""
    text = text.strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def slugify(text: str) -> str:
    """
    Convierte un texto en un identificador normalizado.

    Esta función se conserva como utilidad para posibles nombres de archivo
    o identificadores legibles, aunque los chunk_id actuales son numéricos.
    """
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def create_chunk(
    content: str,
    section: str,
    subsection: str,
    source_file: str,
    chunk_id: str,
    parent_chunk: str | None,
) -> dict[str, Any]:
    """Construye un chunk con el formato esperado por el sistema RAG."""
    clean_content = normalize_text(content)

    chunk: dict[str, Any] = {
        "contenido": clean_content,
        "metadatos": {
            "Seccion": section,
            "Subseccion": subsection,
            "Archivo_Origen": source_file,
            "dominio": DOMAIN,
            "chunk_id": chunk_id,
            "parent_chunk": parent_chunk,
        },
    }

    # El texto enriquecido solo se utiliza para vectorizar chunks hijos.
    if parent_chunk is not None:
        chunk["texto_vector"] = (
            f"DOCUMENTO: {section}\n\n"
            f"SECCIÓN:\n{subsection}\n\n"
            f"CONTENIDO:\n{clean_content}"
        )

    return chunk


def extract_document_chunks(
    markdown_path: Path,
    document_number: int,
) -> list[dict[str, Any]]:
    """Extrae el chunk padre y los chunks hijos de un documento Markdown."""
    lines = markdown_path.read_text(encoding="utf-8").splitlines()

    document_title: str | None = None
    current_subsection: str | None = None
    current_content: list[str] = []
    chunks: list[dict[str, Any]] = []

    parent_id = str(document_number)
    subchunk_counter = 0
    parent_created = False

    def create_parent_if_needed() -> None:
        """Crea el chunk padre una única vez, cuando ya existe un título."""
        nonlocal parent_created

        if parent_created or not document_title:
            return

        chunks.append(
            create_chunk(
                content=document_title,
                section=document_title,
                subsection="",
                source_file=markdown_path.name,
                chunk_id=parent_id,
                parent_chunk=None,
            )
        )
        parent_created = True

    def save_current_subchunk() -> None:
        """Guarda la subsección acumulada si contiene texto útil."""
        nonlocal subchunk_counter, current_content

        if not document_title or not current_subsection:
            current_content = []
            return

        content = normalize_text("\n".join(current_content))

        if not content:
            current_content = []
            return

        create_parent_if_needed()
        subchunk_counter += 1

        chunks.append(
            create_chunk(
                content=content,
                section=document_title,
                subsection=current_subsection,
                source_file=markdown_path.name,
                chunk_id=f"{parent_id}.{subchunk_counter}",
                parent_chunk=parent_id,
            )
        )
        current_content = []

    for line in lines:
        stripped_line = line.strip()

        if not stripped_line:
            current_content.append("")
            continue

        if stripped_line.startswith("## ") and not stripped_line.startswith("### "):
            save_current_subchunk()
            document_title = strip_markdown_heading(stripped_line)
            current_subsection = None
            current_content = []
            continue

        if stripped_line.startswith("### "):
            save_current_subchunk()
            current_subsection = strip_markdown_heading(stripped_line)
            current_content = []
            continue

        current_content.append(line)

    save_current_subchunk()

    if not chunks:
        logger.warning(
            "El documento %s no ha generado ningún chunk.",
            markdown_path.name,
        )

    return chunks


def create_chunks_from_directory(
    input_dir: Path,
    output_file: Path,
    first_parent_id: int = FIRST_PARENT_ID,
) -> list[dict[str, Any]]:
    """Genera un único JSON con los chunks de todos los documentos."""
    if not input_dir.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta de Markdown limpio: {input_dir}"
        )

    markdown_files = sorted(input_dir.glob("*.md"))

    if not markdown_files:
        raise FileNotFoundError(
            f"No se encontraron archivos .md en {input_dir}"
        )

    all_chunks: list[dict[str, Any]] = []

    for document_number, markdown_path in enumerate(
        markdown_files,
        start=first_parent_id,
    ):
        document_chunks = extract_document_chunks(
            markdown_path,
            document_number,
        )
        all_chunks.extend(document_chunks)

        logger.info(
            "%s -> documento %s, %s chunks.",
            markdown_path.name,
            document_number,
            len(document_chunks),
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Archivo JSON creado: %s", output_file)
    logger.info("Total de chunks generados: %s", len(all_chunks))

    return all_chunks

# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def main() -> None:
    """Ejecuta la limpieza de Markdown y la creación de chunks."""
    clean_markdown_directory(
        input_dir=RAW_MARKDOWN_DIR,
        output_dir=CLEAN_MARKDOWN_DIR,
    )

    create_chunks_from_directory(
        input_dir=CLEAN_MARKDOWN_DIR,
        output_file=OUTPUT_CHUNKS_FILE,
        first_parent_id=FIRST_PARENT_ID,
    )

    logger.info("Pipeline completado correctamente.")


if __name__ == "__main__":
    main()
