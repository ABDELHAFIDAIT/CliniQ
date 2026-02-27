from __future__ import annotations
import logging
import re
import uuid
import pdfplumber
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from app.core.config import settings
from app.services.classes.chunk import _Chunk


# logger = logging.getLogger(__name__)


# dimension des vecteurs d'embeddings
VECTOR_DIM = 1024

# Regex pour détecter les chapitres
CHAPTER_RE = re.compile(r"^(PÉDIATRIE|MÉDECINE ADULTE|DENTAIRE)$", re.IGNORECASE)

# Regex pour détecter les sections
SECTION_RE = re.compile(r"^[A-ZÀÉÈÊËÎÏÔÙÛÜ][a-zàéèêëîïôùûü]{2,}(?:[\s\-][A-Za-zÀ-ÿ]+)*$")

# Regex pour détecter les sous-sections "CE QU'IL FAUT SAVOIR", "TRAITEMENT A ENVISAGER", "RECOMMANDATIONS"
SUBSECTION_RE = re.compile(
    r"^(CE QU'IL (?:FAUT SAVOIR|FAUT FAIRE|FAUT EXPLIQUER|NE FAUT PAS FAIRE)"
    r"|TRAITEMENT\s+A\s+ENVISAGER.*"
    r"|RECOMMANDATIONS.*)$",
    re.IGNORECASE,
)

# Regex pour extraire les métadonnées de version, validateur et date
META_VERSION   = re.compile(r"Version\s*:\s*(\d+)")
META_VALIDATOR = re.compile(r"Validation\s*:\s*(.+)")
META_DATE      = re.compile(r"Date\s*:\s*(\d{4})")

# Regex pour ignorer les lignes non pertinentes
SKIP_RE = re.compile(r"^(Guide des Protocoles\s*[-–]\s*\d{4}|\d+|Guide\s+des\s+Protocoles)$", re.IGNORECASE)

# Fournir une liste de référence de termes médicaux
CLINICAL_KEYWORDS = [
    "diarrhée", "déshydratation", "sro", "fièvre", "toux", "antibiotique",
    "paracétamol", "samu", "urgence", "abcès", "douleur", "infection",
    "traitement", "allergie", "asthme", "bronchiolite", "convulsion",
]



class IngestService:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.ensure_collection()


    # Assure que la collection Qdrant existe sinon la crée avec la configuration de vecteurs appropriée
    def ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )


    # Point d'entrée principal pour ingérer un PDF : extrait les chunks, génère les embeddings et insère dans Qdrant
    def ingest(self, pdf_path: str, filename: str) -> dict:
        chunks = self.extract_chunks(pdf_path, filename)
        
        if not chunks:
            return {"status": "error", "message": "No chunks extracted", "document": filename}

        # Génération des embeddings pour tous les chunks en une seule passe
        vectors = self.model.encode(
            [c.embedding_text for c in chunks],
            batch_size=32,
            normalize_embeddings=True,
        )

        # Préparation des points à insérer dans Qdrant
        points = [
            PointStruct(
                id=c.chunk_id,
                vector=vectors[i].tolist(),
                payload=self.chunk_payload(c),
            )
            for i, c in enumerate(chunks)
        ]

        # Insertion des points dans Qdrant
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        
        # Retour d'un résumé de l'ingestion
        return {
            "status": "success",
            "document": filename,
            "chunks_created": len(chunks),
            "breakdown": {
                "text":  sum(1 for c in chunks if c.content_type == "text"),
                "table": sum(1 for c in chunks if c.content_type == "table"),
            },
        }


    # Extraction des chunks d'un PDF en respectant la hiérarchie des chapitres, sections et sous-sections, et en traitant les tables de manière distincte
    def extract_chunks(self, pdf_path: str, filename: str) -> list[_Chunk]:
        # Liste pour stocker les chunks extraits
        chunks: list[_Chunk] = []
        
        # État temporaire pour suivre la hiérarchie et les métadonnées pendant l'extraction
        state = dict(
            chapter="", section="", subsection="",
            version="", validator="", date="",
            page_start=1, buffer=[], current_page=1,
        )

        # Ouverture du PDF et itération page par page
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                state["current_page"] = page_num

                # Extraction des tables et création de chunks spécifiques pour les tables  
                for raw_table in (page.extract_tables() or []):
                    # Nettoyage et conversion de la table en markdown pour une meilleure représentation textuelle
                    md = self.table_to_markdown(raw_table)
                    if not md: continue
                    
                    # Avant de créer un chunk pour la table, on flush le buffer de texte accumulé pour éviter de mélanger les contenus
                    self.flush(chunks, state, filename, page_num)
                    
                    # Création d'un chunk pour la table avec les métadonnées actuelles et ajout à la liste des chunks
                    chunks.append(self.build_chunk(md, "table", state, filename, page_num, page_num))

                # Extraction du texte avec respect du layout
                for line in (page.extract_text(layout=True) or "").splitlines():
                    line = line.strip()
                    
                    if not line or SKIP_RE.match(line): continue

                    # Mise à jour des métadonnées si la ligne correspond à un pattern de métadonnées
                    self.parse_metadata(line, state)
                    
                    # Détection du niveau hiérarchique de la ligne (chapitre, section, sous-section ou texte)
                    level = self.detect_level(line)

                    if level in ("chapter", "section", "subsection"):
                        self.flush(chunks, state, filename, page_num)
                        if level == "chapter": state.update(chapter=line, section="", subsection="")
                        elif level == "section": state.update(section=line, subsection="")
                        else: state["subsection"] = line
                        state["page_start"] = page_num
                    else:
                        state["buffer"].append(line)

        # Après la dernière page, on flush le buffer pour capturer tout texte restant
        self.flush(chunks, state, filename, state["current_page"])
        return chunks


    # Convertit une table extraite en markdown pour une meilleure représentation textuelle
    def table_to_markdown(self, table: list[list]) -> str:
        if not table or not table[0]: return ""
        cleaned = [
            [str(cell).replace("\n", " ").strip() if cell else "" for cell in row] 
            for row in table
        ]
        header, rows = cleaned[0], cleaned[1:]
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        lines.extend(["| " + " | ".join(r + [""] * (len(header) - len(r))) + " |" for r in rows])
        return "\n".join(lines)


    # Flush le buffer de texte accumulé pour créer des chunks de texte avant d'ajouter une nouvelle section ou une table
    def flush(self, chunks: list, state: dict, filename: str, page_end: int) -> None:
        text = "\n".join(state["buffer"]).strip()
        state["buffer"] = []
        if not text: return
        # Division du texte en chunks respectant la limite de mots, avec un chevauchement pour maintenir le contexte, et création de chunks pour chaque partie
        for part in self.split_text(text):
            chunks.append(self.build_chunk(part, "text", state, filename, state["page_start"], page_end))


    # Construction d'un chunk à partir du contenu, du type de contenu, de l'état actuel de la hiérarchie et des métadonnées, et du contexte de pagination
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


    # Divise un long texte en plusieurs parties respectant la limite de mots, avec un chevauchement pour maintenir le contexte entre les chunks
    def split_text(self, text: str) -> list[str]:
        # Si le texte est déjà court, on le retourne tel quel
        if len(text.split()) <= settings.CHUNK_SIZE: return [text]
        
        # Utilisation d'une approche basée sur les phrases pour diviser le texte, en respectant les limites de mots et en assurant un chevauchement pour maintenir le contexte
        sentences = re.split(r'(?<=[.!?•\n])\s+', text)
        
        # Construction des chunks en accumulant les phrases jusqu'à atteindre la limite de mots, puis en créant un nouveau chunk avec un chevauchement pour les phrases suivantes
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


    # Détecte le niveau hiérarchique d'une ligne (chapitre, section, sous-section ou texte) en utilisant des regex spécifiques pour chaque niveau
    def detect_level(self, line: str) -> str:
        if CHAPTER_RE.match(line): return "chapter"
        if SUBSECTION_RE.match(line): return "subsection"
        if SECTION_RE.match(line) and len(line) < 60: return "section"
        return "text"


    # Parse les métadonnées de version, validateur et date à partir d'une ligne de texte en utilisant des regex spécifiques, et met à jour l'état avec les valeurs extraites
    def parse_metadata(self, line: str, state: dict) -> None:
        for key, pattern in (("version", META_VERSION), ("validator", META_VALIDATOR), ("date", META_DATE)):
            m = pattern.search(line)
            if m: state[key] = m.group(1).strip()


    # Prépare le payload à insérer dans Qdrant pour un chunk donné, en incluant toutes les métadonnées pertinentes et le contenu du chunk
    def chunk_payload(self, c: _Chunk) -> dict:
        return {
            "document": c.document, "page_start": c.page_start, "page_end": c.page_end,
            "chapter": c.chapter, "section": c.section, "subsection": c.subsection,
            "content_type": c.content_type, "patient_population": c.patient_population,
            "clinical_tags": c.clinical_tags, "version": c.version,
            "validated_by": c.validated_by, "date": c.date, "content": c.content,
        }


    # 
    def infer_population(self, chapter: str) -> str:
        ch = chapter.lower()
        if "pédiatrie" in ch: return "pediatrie"
        if "adulte" in ch: return "adulte"
        if "dentaire" in ch: return "dentaire"
        return "general"


    # 
    def extract_tags(self, text: str) -> list[str]:
        tl = text.lower()
        return [kw for kw in CLINICAL_KEYWORDS if kw in tl]




ingest_service = IngestService()