# expected_parent_ids reflect the consolidated table-aware pipeline's
# numbering in data/financial_intelligence.db (re-derived 2026-09-15 after
# merging v3's extraction into the primary pipeline — see SPEC.md's
# consolidation section). Verified by direct SQL inspection of
# parent_documents, same technique as every prior re-derivation; reconcile
# again if the DB is ever wiped and re-ingested.
EVAL_QUESTIONS = [
    {"question": "What was CBA's statutory net profit after tax for the 2026 half year?", "expected_parent_ids": {227}},
    {"question": "What interim dividend per share did CBA declare for 1H26?", "expected_parent_ids": {145, 154, 170, 171}},
    {"question": "What was CBA's pre-provision profit for the half year?", "expected_parent_ids": {153}},
    {"question": "What was CBA's net interest margin (NIM) for 1H26?", "expected_parent_ids": {228}},
    {"question": "What was CBA's Common Equity Tier 1 (CET1) capital ratio?", "expected_parent_ids": {163, 230}},
    {"question": "What was CBA's return on equity (ROE) for the half?", "expected_parent_ids": {170}},
    {"question": "What was CBA's net stable funding ratio (NSFR)?", "expected_parent_ids": {167}},
    {"question": "What was CBA's loan impairment expense for the half?", "expected_parent_ids": {229}},
]
