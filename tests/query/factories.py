from ingestion.schemas import Clause, ClauseMetadata, PrivilegeLevel


def make_clause(
    clause_id: str,
    text: str = "Some clause text.",
    section_id: str | None = None,
    heading: str | None = None,
    start: int = 0,
    end: int | None = None,
) -> Clause:
    return Clause(
        clause_id=clause_id,
        section_id=section_id,
        heading=heading,
        text=text,
        start_offset=start,
        end_offset=end if end is not None else start + len(text),
    )


def make_metadata(
    tenant_id: str = "tenant-1",
    matter_id: str = "matter-1",
    jurisdiction: str | None = None,
    governing_law: str | None = None,
    privilege_level: PrivilegeLevel = PrivilegeLevel.CONFIDENTIAL,
) -> ClauseMetadata:
    return ClauseMetadata(
        tenant_id=tenant_id,
        matter_id=matter_id,
        jurisdiction=jurisdiction,
        governing_law=governing_law,
        privilege_level=privilege_level,
    )
