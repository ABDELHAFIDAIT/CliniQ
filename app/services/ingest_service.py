# from __future__ import annotations
# import logging
# import re
# import uuid
# from pathlib import Path
# from typing import Optional
# import pdfplumber
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, PointStruct, VectorParams
# from app.core.config import settings
# from app.services.classes.chunk import _Chunk



# logger = logging.getLogger(__name__)


# VECTOR_DIM = 1024

# _CHAPTER_RE = re.compile(
#     r"^(PÉDIATRIE|MÉDECINE ADULTE|DENTAIRE)$", 
#     re.IGNORECASE
# )

# _SECTION_RE = re.compile(
#     r"^[A-ZÀÉÈÊËÎÏÔÙÛÜ][a-zàéèêëîïôùûü]{2,}(?:[\s\-][A-Za-zÀ-ÿ]+)*$"
# )

# _SUBSECTION_RE = re.compile(
#     r"^(CE QU'IL (?:FAUT SAVOIR|FAUT FAIRE|FAUT EXPLIQUER|NE FAUT PAS FAIRE)"
#     r"|TRAITEMENT\s+A\s+ENVISAGER.*"
#     r"|RECOMMANDATIONS.*)$",
#     re.IGNORECASE,
# )


# _META_VERSION   = re.compile(r"Version\s*:\s*(\d+)")
# _META_VALIDATOR = re.compile(r"Validation\s*:\s*(.+)")
# _META_DATE      = re.compile(r"Date\s*:\s*(\d{4})")


# _CLINICAL_KEYWORDS = [
#     "diarrhée", "déshydratation", "sro", "fièvre", "toux", "antibiotique",
#     "paracétamol", "samu", "urgence", "abcès", "douleur", "infection",
#     "traitement", "allergie", "asthme", "bronchiolite", "convulsion",
# ]

# _SKIP_RE = re.compile(
#     r"^(Guide des Protocoles\s*[-–]\s*\d{4}|\d+|Guide\s+des\s+Protocoles)$",
#     re.IGNORECASE,
# )




# class IngestService:
#     def __init__(self):
#         logger.info(f"Loading Embedding Model: {settings.EMBEDDING_MODEL}")
#         self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
#         logger.info(f"Connecting to Qdrant : {settings.QDRANT_URL}")
#         self.client = QdrantClient(url=settings.QDRANT_URL)
#         self.collection_name = settings.QDRANT_COLLECTION_NAME
        
#         self.ensure_collection()
#         logger.info("IngestService ready to process documents !")
    
    
    
#     # ============================
#     #   Main Ingestion Pipeline
#     # ============================
#     def ingest(self, pdf_path: str, filename: str) -> dict:
#         logger.info("Ingesting '%s'", filename)

#         chunks = self.extract_chunks(pdf_path, filename)
#         if not chunks:
#             return {"status": "error", "message": "No chunks extracted", "document": filename}

#         vectors = self.model.encode(
#             [c.embedding_text for c in chunks],
#             batch_size=32,
#             show_progress_bar=False,
#             normalize_embeddings=True,
#         )

#         points = [
#             PointStruct(
#                 id=c.chunk_id,
#                 vector=vectors[i].tolist(),
#                 payload=self.chunk_payload(c),
#             )
#             for i, c in enumerate(chunks)
#         ]

#         self.client.upsert(
#             collection_name=self.collection_name,
#             points=points,
#             wait=True,
#         )

#         logger.info("Upserted %d points for '%s'", len(points), filename)
#         return {
#             "status":         "success",
#             "document":       filename,
#             "chunks_created": len(chunks),
#             "collection":     self.collection_name,
#             "breakdown": {
#                 "text":  sum(1 for c in chunks if c.content_type == "text"),
#                 "table": sum(1 for c in chunks if c.content_type == "table"),
#             },
#         }
    
    
    
    
#     # =================================
#     #  PDF Parsing & Chunk Extraction
#     # =================================
#     def extract_chunks(self, pdf_path: str, filename: str) -> list[_Chunk]:
#         chunks: list[_Chunk] = []

#         state = dict(
#             chapter="", section="", subsection="",
#             version="", validator="", date="",
#             page_start=1, buffer=[], current_page=1,
#         )

#         with pdfplumber.open(pdf_path) as pdf:
#             for page_num, page in enumerate(pdf.pages, start=1):
#                 state["current_page"] = page_num

#                 # ── Tables first ──────────────────
#                 for raw_table in (page.extract_tables() or []):
#                     md = self.table_to_markdown(raw_table)
                    
#                     if not md:
#                         continue
                    
#                     self.flush(chunks, state, filename, page_num)
                    
#                     chunks.append(self.build_chunk(
#                         content=md, ctype="table",
#                         state=state, filename=filename,
#                         p_start=page_num, p_end=page_num,
#                     ))

#                 # ── Text lines ────────────────────
#                 for line in (page.extract_text(layout=True) or "").splitlines():
#                     line = line.strip()
#                     if not line or _SKIP_RE.match(line):
#                         continue

#                     self.parse_metadata(line, state)
#                     level = self.detect_level(line)

#                     if level in ("chapter", "section", "subsection"):
#                         self.flush(chunks, state, filename, page_num)
#                         if level == "chapter":
#                             state.update(chapter=line, section="", subsection="")
#                         elif level == "section":
#                             state.update(section=line, subsection="")
#                         else:
#                             state["subsection"] = line
#                         state["page_start"] = page_num
#                         state["buffer"] = []
#                     else:
#                         state["buffer"].append(line)

#         self.flush(chunks, state, filename, state["current_page"])
#         return chunks
    
    
    
    
#     # ============================
#     #  Ensure Qdrant Collection
#     # ============================
#     def ensure_collection(self):
#         if not self.client.collection_exists(self.collection_name):
#             logger.info(f"Creating Qdrant collection: {self.collection_name}")
#             self.client.create_collection(
#                 collection_name=self.collection_name,
#                 vectors_config=VectorParams(
#                     size=VECTOR_DIM, 
#                     distance=Distance.COSINE
#                 ),
#             )
    
    
    
    
#     # ================================
#     #  Table to Markdown Conversion
#     # ================================
#     def table_to_markdown(table: list[list]) -> str:
#         if not table or not table[0]:
#             return ""
        
#         cleaned = [
#             [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
#             for row in table
#         ]
        
#         header = cleaned[0]
#         sep    = ["---"] * len(header)
#         rows   = [r + [""] * (len(header) - len(r)) for r in cleaned[1:]]
        
#         lines  = (
#             ["| " + " | ".join(header) + " |"]
#             + ["| " + " | ".join(sep) + " |"]
#             + ["| " + " | ".join(r[:len(header)]) + " |" for r in rows]
#         )
        
#         return "\n".join(lines)
    
    
    
    
#     # ===================================
#     #  Text Buffer Flushing & Chunking
#     # ===================================
#     def flush(self, chunks: list, state: dict, filename: str, page_end: int) -> None:
#         text = "\n".join(state["buffer"]).strip()
#         state["buffer"] = []
        
#         if not text:
#             return
        
#         for part in self.split_text(text):
#             chunks.append(self.build_chunk(
#                 content=part, ctype="text",
#                 state=state, filename=filename,
#                 p_start=state["page_start"], p_end=page_end,
#             ))
    
    
    
    
#     # ===================================
#     #  Chunk Building with Metadata
#     # ===================================
#     def build_chunk(self, content: str, ctype: str, state: dict, filename: str, p_start: int, p_end: int) -> _Chunk:
#         prefix = (
#             f"Document: {filename}. "
#             f"Chapitre: {state['chapter']}. "
#             f"Section: {state['section']}. "
#             f"Sous-section: {state['subsection']}. "
#         )
#         return _Chunk(
#             chunk_id=str(uuid.uuid4()),
#             document=filename,
#             page_start=p_start,
#             page_end=p_end,
#             chapter=state["chapter"],
#             section=state["section"],
#             subsection=state["subsection"],
#             content_type=ctype,
#             patient_population=self.infer_population(state["chapter"]),
#             clinical_tags=self.extract_tags(content),
#             version=state["version"],
#             validated_by=state["validator"],
#             date=state["date"],
#             content=content,
#             embedding_text=prefix + content,
#         )
        
        


#     # ===================================
#     #  Text Splitting with Overlap
#     # ===================================
#     def split_text(self, text: str) -> list[str]:
#         if len(text.split()) <= settings.CHUNK_SIZE:
#             return [text]

#         sentences = re.split(r'(?<=[.!?•\n])\s+', text)
#         parts: list[str] = []
#         current: list[str] = []
#         current_len = 0

#         for sent in sentences:
#             sent_len = len(sent.split())
#             if current_len + sent_len > settings.CHUNK_SIZE and current:
#                 parts.append(" ".join(current))
#                 tail = " ".join(current).split()[-settings.CHUNK_OVERLAP:]
#                 current = [" ".join(tail), sent] if tail else [sent]
#                 current_len = len(current[0].split()) + sent_len
#             else:
#                 current.append(sent)
#                 current_len += sent_len

#         if current:
#             parts.append(" ".join(current))
#         return parts
    
    
    
    
    
#     # =====================================
#     #  Metadata Parsing & Level Detection
#     # =====================================
#     def detect_level(line: str) -> str:
#         if _CHAPTER_RE.match(line):
#             return "chapter"
#         if _SUBSECTION_RE.match(line):
#             return "subsection"
#         if _SECTION_RE.match(line) and len(line) < 60:
#             return "section"
#         return "text"
    
    
    
    
    
    
#     # ======================================
#     #  Metadata Parsing & Level Detection
#     # ======================================
#     def parse_metadata(line: str, state: dict) -> None:
#         for key, pattern in (
#             ("version",   _META_VERSION),
#             ("validator", _META_VALIDATOR),
#             ("date",      _META_DATE),
#         ):
#             m = pattern.search(line)
#             if m:
#                 state[key] = m.group(1).strip()
    
    
    
    
#     # ======================================
#     #  Chunk Payload Preparation for Qdrant
#     # ======================================
#     def chunk_payload(c: _Chunk) -> dict:
#         return {
#             "document":           c.document,
#             "page_start":         c.page_start,
#             "page_end":           c.page_end,
#             "chapter":            c.chapter,
#             "section":            c.section,
#             "subsection":         c.subsection,
#             "content_type":       c.content_type,
#             "patient_population": c.patient_population,
#             "clinical_tags":      c.clinical_tags,
#             "version":            c.version,
#             "validated_by":       c.validated_by,
#             "date":               c.date,
#             "content":            c.content,
#         }
    
    
    
    
#     # =============================================
#     #  Infer Patient Population from Chapter Title
#     # =============================================
#     def infer_population(chapter: str) -> str:
#         ch = chapter.lower()
#         if "pédiatrie" in ch:  return "pediatrie"
#         if "adulte"    in ch:  return "adulte"
#         if "dentaire"  in ch:  return "dentaire"
#         return "general"





#     # =============================================
#     #  Clinical Tag Extraction from Text Content
#     # =============================================
#     def extract_tags(text: str) -> list[str]:
#         tl = text.lower()
#         return [kw for kw in _CLINICAL_KEYWORDS if kw in tl]    
    



# ingest_service = IngestService()






























from __future__ import annotations
import logging
import re
import uuid
from pathlib import Path
from typing import Optional
import pdfplumber
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from app.core.config import settings
from app.services.classes.chunk import _Chunk

logger = logging.getLogger(__name__)

VECTOR_DIM = 1024

# Regex de structure basées sur le Guide des Protocoles 2025
_CHAPTER_RE = re.compile(r"^(PÉDIATRIE|MÉDECINE ADULTE|DENTAIRE)$", re.IGNORECASE)
_SECTION_RE = re.compile(r"^[A-ZÀÉÈÊËÎÏÔÙÛÜ][a-zàéèêëîïôùûü]{2,}(?:[\s\-][A-Za-zÀ-ÿ]+)*$")
_SUBSECTION_RE = re.compile(
    r"^(CE QU'IL (?:FAUT SAVOIR|FAUT FAIRE|FAUT EXPLIQUER|NE FAUT PAS FAIRE)"
    r"|TRAITEMENT\s+A\s+ENVISAGER.*"
    r"|RECOMMANDATIONS.*)$",
    re.IGNORECASE,
)

_META_VERSION   = re.compile(r"Version\s*:\s*(\d+)")
_META_VALIDATOR = re.compile(r"Validation\s*:\s*(.+)")
_META_DATE      = re.compile(r"Date\s*:\s*(\d{4})")

_SKIP_RE = re.compile(r"^(Guide des Protocoles\s*[-–]\s*\d{4}|\d+|Guide\s+des\s+Protocoles)$", re.IGNORECASE)

_CLINICAL_KEYWORDS = [
    "diarrhée", "déshydratation", "sro", "fièvre", "toux", "antibiotique",
    "paracétamol", "samu", "urgence", "abcès", "douleur", "infection",
    "traitement", "allergie", "asthme", "bronchiolite", "convulsion",
]

class IngestService:
    def __init__(self):
        logger.info(f"Loading Embedding Model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        logger.info(f"Connecting to Qdrant : {settings.QDRANT_URL}")
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        
        self.ensure_collection()
        logger.info("IngestService ready to process documents !")

    def ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            logger.info(f"Creating Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

    def ingest(self, pdf_path: str, filename: str) -> dict:
        logger.info("Ingesting '%s'", filename)
        chunks = self.extract_chunks(pdf_path, filename)
        
        if not chunks:
            return {"status": "error", "message": "No chunks extracted", "document": filename}

        vectors = self.model.encode(
            [c.embedding_text for c in chunks],
            batch_size=32,
            normalize_embeddings=True,
        )

        points = [
            PointStruct(
                id=c.chunk_id,
                vector=vectors[i].tolist(),
                payload=self.chunk_payload(c),
            )
            for i, c in enumerate(chunks)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return {
            "status": "success",
            "document": filename,
            "chunks_created": len(chunks),
            "breakdown": {
                "text":  sum(1 for c in chunks if c.content_type == "text"),
                "table": sum(1 for c in chunks if c.content_type == "table"),
            },
        }

    def extract_chunks(self, pdf_path: str, filename: str) -> list[_Chunk]:
        chunks: list[_Chunk] = []
        state = dict(
            chapter="", section="", subsection="",
            version="", validator="", date="",
            page_start=1, buffer=[], current_page=1,
        )

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                state["current_page"] = page_num

                # 1. Extraction des tables (Scores de Silverman, Arbres décisionnels, etc.)
                for raw_table in (page.extract_tables() or []):
                    md = self.table_to_markdown(raw_table)
                    if not md: continue
                    self.flush(chunks, state, filename, page_num)
                    chunks.append(self.build_chunk(md, "table", state, filename, page_num, page_num))

                # 2. Extraction du texte avec respect du layout
                for line in (page.extract_text(layout=True) or "").splitlines():
                    line = line.strip()
                    if not line or _SKIP_RE.match(line): continue

                    self.parse_metadata(line, state)
                    level = self.detect_level(line)

                    if level in ("chapter", "section", "subsection"):
                        self.flush(chunks, state, filename, page_num)
                        if level == "chapter": state.update(chapter=line, section="", subsection="")
                        elif level == "section": state.update(section=line, subsection="")
                        else: state["subsection"] = line
                        state["page_start"] = page_num
                    else:
                        state["buffer"].append(line)

        self.flush(chunks, state, filename, state["current_page"])
        return chunks

    def table_to_markdown(self, table: list[list]) -> str:
        if not table or not table[0]: return ""
        cleaned = [[str(cell).replace("\n", " ").strip() if cell else "" for cell in row] for row in table]
        header, rows = cleaned[0], cleaned[1:]
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        lines.extend(["| " + " | ".join(r + [""] * (len(header) - len(r))) + " |" for r in rows])
        return "\n".join(lines)

    def flush(self, chunks: list, state: dict, filename: str, page_end: int) -> None:
        text = "\n".join(state["buffer"]).strip()
        state["buffer"] = []
        if not text: return
        for part in self.split_text(text):
            chunks.append(self.build_chunk(part, "text", state, filename, state["page_start"], page_end))

    def build_chunk(self, content: str, ctype: str, state: dict, filename: str, p_start: int, p_end: int) -> _Chunk:
        prefix = f"Doc: {filename}. Chapitre: {state['chapter']}. Section: {state['section']}. "
        return _Chunk(
            chunk_id=str(uuid.uuid4()),
            document=filename,
            page_start=p_start,
            page_end=p_end,
            chapter=state["chapter"],
            section=state["section"],
            subsection=state["subsection"],
            content_type=ctype,
            patient_population=self.infer_population(state["chapter"]),
            clinical_tags=self.extract_tags(content),
            version=state["version"],
            validated_by=state["validator"],
            date=state["date"],
            content=content,
            embedding_text=prefix + content,
        )

    def split_text(self, text: str) -> list[str]:
        # Utilisation des paramètres de config pour le découpage récursif
        if len(text.split()) <= settings.CHUNK_SIZE: return [text]
        sentences = re.split(r'(?<=[.!?•\n])\s+', text)
        parts, current, current_len = [], [], 0
        for sent in sentences:
            length = len(sent.split())
            if current_len + length > settings.CHUNK_SIZE and current:
                parts.append(" ".join(current))
                current = current[-max(1, settings.CHUNK_OVERLAP // 10):] + [sent]
                current_len = sum(len(s.split()) for s in current)
            else:
                current.append(sent)
                current_len += length
        if current: parts.append(" ".join(current))
        return parts

    def detect_level(self, line: str) -> str:
        if _CHAPTER_RE.match(line): return "chapter"
        if _SUBSECTION_RE.match(line): return "subsection"
        if _SECTION_RE.match(line) and len(line) < 60: return "section"
        return "text"

    def parse_metadata(self, line: str, state: dict) -> None:
        for key, pattern in (("version", _META_VERSION), ("validator", _META_VALIDATOR), ("date", _META_DATE)):
            m = pattern.search(line)
            if m: state[key] = m.group(1).strip()

    def chunk_payload(self, c: _Chunk) -> dict:
        return {
            "document": c.document, "page_start": c.page_start, "page_end": c.page_end,
            "chapter": c.chapter, "section": c.section, "subsection": c.subsection,
            "content_type": c.content_type, "patient_population": c.patient_population,
            "clinical_tags": c.clinical_tags, "version": c.version,
            "validated_by": c.validated_by, "date": c.date, "content": c.content,
        }

    def infer_population(self, chapter: str) -> str:
        ch = chapter.lower()
        if "pédiatrie" in ch: return "pediatrie"
        if "adulte" in ch: return "adulte"
        if "dentaire" in ch: return "dentaire"
        return "general"

    def extract_tags(self, text: str) -> list[str]:
        tl = text.lower()
        return [kw for kw in _CLINICAL_KEYWORDS if kw in tl]

ingest_service = IngestService()