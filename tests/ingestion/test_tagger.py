from datetime import date

from ingestion.metadata.tagger import tag_clause
from ingestion.schemas import Clause, PrivilegeLevel


class FakeLLMClient:
    def __init__(self, response: dict):
        self._response = response
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self._response


def _sample_clause() -> Clause:
    return Clause(
        clause_id="doc-1::clause-006",
        section_id="4",
        heading="Governing Law and Jurisdiction",
        text=(
            "This Agreement shall be governed by the laws of the State of "
            "Delaware. The parties submit to the exclusive jurisdiction of "
            "the courts of New York."
        ),
        start_offset=0,
        end_offset=100,
    )


def test_jurisdiction_and_governing_law_land_in_separate_fields():
    llm_client = FakeLLMClient(
        {
            "contract_type": "Master Services Agreement",
            "clause_type": "Governing Law",
            "parties": ["Acme Corp", "Globex LLC"],
            "effective_date": "2024-01-15",
            "signature_date": None,
            "jurisdiction": "New York",
            "governing_law": "Delaware",
            "contains_pii": False,
        }
    )

    metadata = tag_clause(
        _sample_clause(),
        tenant_id="tenant-1",
        matter_id="matter-1",
        privilege_level=PrivilegeLevel.CONFIDENTIAL,
        llm_client=llm_client,
    )

    assert metadata.jurisdiction == "New York"
    assert metadata.governing_law == "Delaware"
    assert metadata.jurisdiction != metadata.governing_law
    assert metadata.effective_date == date(2024, 1, 15)


def test_tenant_and_matter_id_come_from_caller_not_llm_response():
    """Security-critical: even if the LLM response tried to smuggle in a
    tenant/matter id (e.g. via prompt injection from counterparty-authored
    contract text), the caller-supplied values must win. See CLAUDE.md #7 and
    the docstring on ClauseMetadata."""
    llm_client = FakeLLMClient(
        {
            "contract_type": "NDA",
            "clause_type": "Confidentiality",
            "parties": [],
            "effective_date": None,
            "signature_date": None,
            "jurisdiction": None,
            "governing_law": None,
            "contains_pii": False,
            "tenant_id": "attacker-tenant",
            "matter_id": "attacker-matter",
        }
    )

    metadata = tag_clause(
        _sample_clause(),
        tenant_id="tenant-1",
        matter_id="matter-1",
        privilege_level=PrivilegeLevel.PRIVILEGED,
        llm_client=llm_client,
    )

    assert metadata.tenant_id == "tenant-1"
    assert metadata.matter_id == "matter-1"
    assert metadata.privilege_level == PrivilegeLevel.PRIVILEGED


def test_missing_optional_fields_default_safely():
    llm_client = FakeLLMClient({})

    metadata = tag_clause(
        _sample_clause(),
        tenant_id="tenant-1",
        matter_id="matter-1",
        privilege_level=PrivilegeLevel.CONFIDENTIAL,
        llm_client=llm_client,
    )

    assert metadata.parties == []
    assert metadata.contains_pii is False
    assert metadata.jurisdiction is None
    assert metadata.governing_law is None
