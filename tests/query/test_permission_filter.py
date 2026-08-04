import pytest

from query.filters.permission import PermissionViolation, assert_permission, build_permission_filter
from query.schemas import PermissionContext, RetrievedClause
from tests.query.factories import make_clause, make_metadata


def test_build_permission_filter_includes_both_tenant_and_matter():
    context = PermissionContext(tenant_id="tenant-1", matter_id="matter-1")
    conditions = build_permission_filter(context)

    fields = {c.field: c.value for c in conditions}
    assert fields == {"tenant_id": "tenant-1", "matter_id": "matter-1"}


def test_assert_permission_passes_when_all_clauses_match_context():
    context = PermissionContext(tenant_id="tenant-1", matter_id="matter-1")
    retrieved = [
        RetrievedClause(
            clause=make_clause("doc::c1"),
            metadata=make_metadata(tenant_id="tenant-1", matter_id="matter-1"),
            score=1.0,
            source="vector",
        )
    ]
    assert_permission(retrieved, context)  # no raise


def test_assert_permission_raises_on_cross_tenant_leak():
    """The core CLAUDE.md #7 non-negotiable: tested with two isolated
    tenants -- a clause belonging to tenant-2 must never be returned for a
    tenant-1 query, and if it somehow is, this must fail loudly."""
    context = PermissionContext(tenant_id="tenant-1", matter_id="matter-1")
    leaked = RetrievedClause(
        clause=make_clause("doc::c1"),
        metadata=make_metadata(tenant_id="tenant-2", matter_id="matter-1"),
        score=1.0,
        source="vector",
    )

    with pytest.raises(PermissionViolation):
        assert_permission([leaked], context)


def test_assert_permission_raises_on_cross_matter_leak_within_same_tenant():
    context = PermissionContext(tenant_id="tenant-1", matter_id="matter-A")
    leaked = RetrievedClause(
        clause=make_clause("doc::c1"),
        metadata=make_metadata(tenant_id="tenant-1", matter_id="matter-B"),
        score=1.0,
        source="vector",
    )

    with pytest.raises(PermissionViolation):
        assert_permission([leaked], context)
