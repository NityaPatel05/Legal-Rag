"""R3: the app's runtime passage size and the library default are one decision."""

from ingestion.config import Settings
from ingestion.parsing.clause_parser import DEFAULT_MAX_PASSAGE_CHARS


def test_runtime_and_library_passage_size_agree():
    settings = Settings(_env_file=None)
    assert settings.max_passage_chars == DEFAULT_MAX_PASSAGE_CHARS == 1000
    assert settings.generation_top_k == 10
