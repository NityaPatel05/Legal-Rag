"""Controlled vocabularies for clause_type and contract_type (roadmap R7).

Free-text LLM labels drift ("Term and Termination", "Termination Rights",
"termination") and can't be filtered, counted or evaluated against CUAD. The
tagger prompt now lists the allowed labels, and these functions map whatever
comes back onto them deterministically -- the prompt reduces drift, the code
guarantees the stored value is in the vocabulary.

clause_type = CUAD's 41 label categories (Hendrycks et al. 2021, the ids in
CUAD_v1.json) + general clause types. CUAD's categories are *review flags*,
not a clause taxonomy: several are document-level (Parties, Agreement Date)
and the most common contract clauses (confidentiality, indemnification,
payment, definitions) have no CUAD category at all. Mapping onto CUAD's 41
alone would label most of a contract "Other"; the general types keep that
information while CUAD-category clauses stay comparable with CUAD.

contract_type = CUAD's 25 contract types + the common types CUAD lacks (NDA,
employment, purchase, lease, loan). Mapping is by ordered keyword rules --
checked against the 510 CUAD titles (see tests) -- and anything unmatched is
"Other" rather than a guess."""

from __future__ import annotations

import re

CUAD_CLAUSE_CATEGORIES = (
    "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date",
    "Renewal Term", "Notice Period To Terminate Renewal", "Governing Law",
    "Most Favored Nation", "Non-Compete", "Exclusivity", "No-Solicit Of Customers",
    "Competitive Restriction Exception", "No-Solicit Of Employees", "Non-Disparagement",
    "Termination For Convenience", "Rofr/Rofo/Rofn", "Change Of Control", "Anti-Assignment",
    "Revenue/Profit Sharing", "Price Restrictions", "Minimum Commitment", "Volume Restriction",
    "Ip Ownership Assignment", "Joint Ip Ownership", "License Grant", "Non-Transferable License",
    "Affiliate License-Licensor", "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License", "Irrevocable Or Perpetual License",
    "Source Code Escrow", "Post-Termination Services", "Audit Rights", "Uncapped Liability",
    "Cap On Liability", "Liquidated Damages", "Warranty Duration", "Insurance",
    "Covenant Not To Sue", "Third Party Beneficiary",
)

GENERAL_CLAUSE_TYPES = (
    "Definitions", "Term", "Termination", "Payment", "Confidentiality", "Indemnification",
    "Representations And Warranties", "Limitation Of Liability", "Intellectual Property",
    "Dispute Resolution", "Jurisdiction", "Notices", "Force Majeure", "Assignment",
    "Amendment", "Entire Agreement", "Severability", "Waiver", "Compliance With Laws",
    "Data Protection", "Obligations", "Signature", "Other",
)

CLAUSE_TYPES = CUAD_CLAUSE_CATEGORIES + GENERAL_CLAUSE_TYPES

CUAD_CONTRACT_TYPES = (
    "Affiliate", "Agency", "Collaboration", "Co-Branding", "Consulting", "Development",
    "Distributor", "Endorsement", "Franchise", "Hosting", "IP", "Joint Venture", "License",
    "Maintenance", "Manufacturing", "Marketing", "Non-Compete", "Outsourcing", "Promotion",
    "Reseller", "Service", "Sponsorship", "Strategic Alliance", "Supply", "Transportation",
)

CONTRACT_TYPES = CUAD_CONTRACT_TYPES + (
    "Non-Disclosure", "Employment", "Purchase", "Lease", "Loan", "Other",
)


def _key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


_CLAUSE_BY_KEY = {_key(label): label for label in CLAUSE_TYPES}

# Frequent LLM phrasings that aren't a vocabulary label verbatim.
_CLAUSE_ALIASES = {
    "term and termination": "Termination",
    "termination rights": "Termination",
    "termination for cause": "Termination",
    "indemnity": "Indemnification",
    "indemnities": "Indemnification",
    "non disclosure": "Confidentiality",
    "confidential information": "Confidentiality",
    "limitation of liabilities": "Limitation Of Liability",
    "liability": "Limitation Of Liability",
    "warranties": "Representations And Warranties",
    "representations": "Representations And Warranties",
    "warranty": "Representations And Warranties",
    "fees": "Payment",
    "payment terms": "Payment",
    "compensation": "Payment",
    "arbitration": "Dispute Resolution",
    "venue": "Jurisdiction",
    "choice of law": "Governing Law",
    "applicable law": "Governing Law",
    "assignment": "Assignment",
    "non assignment": "Anti-Assignment",
    "non solicitation": "No-Solicit Of Employees",
    "non competition": "Non-Compete",
    "noncompete": "Non-Compete",
    "exclusive": "Exclusivity",
    "most favored customer": "Most Favored Nation",
    "mfn": "Most Favored Nation",
    "license": "License Grant",
    "grant of license": "License Grant",
    "ip ownership": "Ip Ownership Assignment",
    "intellectual property rights": "Intellectual Property",
    "audit": "Audit Rights",
    "records and audit": "Audit Rights",
    "renewal": "Renewal Term",
    "privacy": "Data Protection",
    "miscellaneous": "Other",
    "general": "Other",
    "general provisions": "Other",
    "recitals": "Other",
    "preamble": "Parties",
}


def normalize_clause_type(value: str | None) -> str | None:
    """Maps an LLM clause label onto CLAUSE_TYPES. None stays None (the
    tagger didn't classify it); anything unrecognised becomes "Other"."""
    if value is None or not str(value).strip():
        return None
    key = _key(str(value))
    if key in _CLAUSE_BY_KEY:
        return _CLAUSE_BY_KEY[key]
    if key in _CLAUSE_ALIASES:
        return _CLAUSE_ALIASES[key]
    # "Termination Clause", "Governing Law Provision"
    stripped = re.sub(r"\b(clause|provision|provisions|section)s?\b", "", key).strip()
    stripped = re.sub(r"\s+", " ", stripped)
    return _CLAUSE_BY_KEY.get(stripped) or _CLAUSE_ALIASES.get(stripped) or "Other"


# Ordered: the first matching rule wins. Specific before generic ("web site
# hosting" before "service"; "sub-license" / "license" after "reseller" and
# "distribut" because a distribution agreement usually also grants a license).
_CONTRACT_RULES: tuple[tuple[str, str], ...] = (
    (r"non[- ]?disclosure|confidentiality agreement|\bnda\b|secrecy", "Non-Disclosure"),
    (r"non[- ]?compet|non[- ]?solicit|non[- ]?disparage", "Non-Compete"),
    (r"employment|executive agreement|offer letter", "Employment"),
    (r"joint venture|\bjv\b", "Joint Venture"),
    (r"strategic alliance|alliance", "Strategic Alliance"),
    (r"co[- ]?branding", "Co-Branding"),
    (r"franchise", "Franchise"),
    (r"hosting", "Hosting"),
    (r"sponsor", "Sponsorship"),
    (r"endorse", "Endorsement"),
    (r"reseller|resale|value added", "Reseller"),
    (r"distribut", "Distributor"),
    (r"outsourc", "Outsourcing"),
    (r"manufactur", "Manufacturing"),
    (r"supply|supplier", "Supply"),
    (r"transport|shipping|carrier|freight", "Transportation"),
    (r"maintenance|support agreement", "Maintenance"),
    (r"consult", "Consulting"),
    (r"development|r ?& ?d|research", "Development"),
    (r"collaborat|cooperation", "Collaboration"),
    (r"promot", "Promotion"),
    (r"marketing|advertis", "Marketing"),
    (r"agency|\bagent\b|representative", "Agency"),
    (r"affiliate", "Affiliate"),
    (r"intellectual property|\bip\b|patent|trademark|copyright|technology transfer", "IP"),
    (r"licen[cs]", "License"),
    (r"service|\bmsa\b|\bsow\b|statement of work", "Service"),
    (r"purchase|sale of|sales agreement|asset", "Purchase"),
    (r"lease|sublease|rental", "Lease"),
    (r"loan|credit|promissory|note purchase", "Loan"),
)
_CONTRACT_PATTERNS = tuple((re.compile(p, re.IGNORECASE), label) for p, label in _CONTRACT_RULES)
_CONTRACT_BY_KEY = {_key(label): label for label in CONTRACT_TYPES}


def normalize_contract_type(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    text = str(value)
    exact = _CONTRACT_BY_KEY.get(_key(text))
    if exact:
        return exact
    for pattern, label in _CONTRACT_PATTERNS:
        if pattern.search(text):
            return label
    return "Other"


def prompt_list(labels: tuple[str, ...]) -> str:
    return ", ".join(f'"{label}"' for label in labels)
