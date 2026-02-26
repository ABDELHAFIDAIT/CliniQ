from dataclasses import dataclass


@dataclass
class _Chunk:
    chunk_id:           str
    document:           str
    page_start:         int
    page_end:           int
    chapter:            str
    section:            str
    subsection:         str
    content_type:       str
    patient_population: str
    clinical_tags:      list[str]
    version:            str
    validated_by:       str
    date:               str
    content:            str
    embedding_text:     str