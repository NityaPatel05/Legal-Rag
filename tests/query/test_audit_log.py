import json

from query.filters.audit_log import AuditEvent, InMemoryAuditLogger, JSONLAuditLogger


def test_in_memory_logger_accumulates_events():
    logger = InMemoryAuditLogger()
    logger.log(AuditEvent(event_type="jailbreak_blocked", tenant_id="tenant-1", matter_id="matter-1"))
    logger.log(AuditEvent(event_type="pii_redacted", tenant_id="tenant-1", matter_id="matter-1"))

    assert [e.event_type for e in logger.events] == ["jailbreak_blocked", "pii_redacted"]


def test_jsonl_logger_appends_one_line_per_event(tmp_path):
    path = tmp_path / "audit.jsonl"
    logger = JSONLAuditLogger(path)

    logger.log(AuditEvent(event_type="jailbreak_blocked", tenant_id="tenant-1", matter_id="matter-1"))
    logger.log(AuditEvent(event_type="citation_stripped", tenant_id="tenant-1", matter_id="matter-1"))

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "jailbreak_blocked"
    assert json.loads(lines[1])["event_type"] == "citation_stripped"


def test_jsonl_logger_never_truncates_existing_events(tmp_path):
    """"Immutable" in practice: opening the logger again and logging more
    events must never wipe out what was already written."""
    path = tmp_path / "audit.jsonl"
    JSONLAuditLogger(path).log(AuditEvent(event_type="jailbreak_blocked"))

    # simulate a new process/session reusing the same log file
    JSONLAuditLogger(path).log(AuditEvent(event_type="pii_redacted"))

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_audit_event_details_round_trip():
    event = AuditEvent(event_type="pii_redacted", details={"finding_kinds": ["ssn", "account_number"]})
    assert event.details["finding_kinds"] == ["ssn", "account_number"]
