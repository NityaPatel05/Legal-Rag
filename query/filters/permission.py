"""Permission-first filtering — CLAUDE.md #7: tenant/matter access check
happens before anything touches the vector store, a security control, not a
data-hygiene step. Two layers, deliberately redundant: build_permission_filter
is passed into the vector search call (pre-retrieval, the primary control);
assert_permission is a post-retrieval check that fails loudly if anything
the filter should have excluded comes back anyway, rather than trusting a
single layer for something security-critical."""

from __future__ import annotations

from query.schemas import FilterCondition, PermissionContext, RetrievedClause


class PermissionViolation(RuntimeError):
    """A retrieved clause's tenant/matter didn't match the requesting
    context. Indicates the pre-retrieval filter failed (bug or misconfigured
    vector store) — not something to recover from silently."""


def build_permission_filter(context: PermissionContext) -> list[FilterCondition]:
    return [
        FilterCondition(field="tenant_id", value=context.tenant_id),
        FilterCondition(field="matter_id", value=context.matter_id),
    ]


def assert_permission(
    retrieved: list[RetrievedClause], context: PermissionContext, document_id: str | None = None
) -> None:
    """With `document_id`, also rejects a clause of another document of the
    same matter: queries are scoped to one document (invariant #9), and a
    clause of a different contract would be shown and cited as this one's."""
    for r in retrieved:
        if r.metadata.tenant_id != context.tenant_id or r.metadata.matter_id != context.matter_id:
            raise PermissionViolation(
                f"Clause {r.clause.clause_id} (tenant={r.metadata.tenant_id}, "
                f"matter={r.metadata.matter_id}) does not match requesting context "
                f"(tenant={context.tenant_id}, matter={context.matter_id})"
            )
        if document_id is not None and r.clause.clause_id.split("::clause-")[0] != document_id:
            raise PermissionViolation(
                f"Clause {r.clause.clause_id} is not part of the queried document {document_id}"
            )
