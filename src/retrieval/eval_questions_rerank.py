# Extended ground-truth set for the two-stage retrieval / rerank eval (§15)
# and the Faithfulness / Context Precision eval (§17). Kept separate from
# eval_questions.py (the original 8) so hyde_eval.py and query_expansion_eval.py's
# already-recorded SPEC.md results stay reproducible against the exact set they
# were measured on.
#
# expected_parent_ids and expected_answer both re-derived 2026-09-15 against
# the consolidated table-aware pipeline's numbering (see SPEC.md's
# consolidation section), verified by direct SQL inspection of
# parent_documents by searching for each fact's distinctive text.
EVAL_QUESTIONS = [
    {"question": "What was CBA's statutory net profit after tax for the 2026 half year?", "expected_parent_ids": {227}, "expected_answer": "$5,412 million"},
    {"question": "What interim dividend per share did CBA declare for 1H26?", "expected_parent_ids": {145, 154, 170, 171}, "expected_answer": "$2.35 per share, fully franked"},
    {"question": "What was CBA's pre-provision profit for the half year?", "expected_parent_ids": {153}, "expected_answer": "$8,131 million"},
    {"question": "What was CBA's net interest margin (NIM) for 1H26?", "expected_parent_ids": {228}, "expected_answer": "2.04%"},
    {"question": "What was CBA's Common Equity Tier 1 (CET1) capital ratio?", "expected_parent_ids": {163, 230}, "expected_answer": "12.3%"},
    {"question": "What was CBA's return on equity (ROE) for the half?", "expected_parent_ids": {170}, "expected_answer": "13.8%"},
    {"question": "What was CBA's net stable funding ratio (NSFR)?", "expected_parent_ids": {167}, "expected_answer": "117%"},
    {"question": "What was CBA's loan impairment expense for the half?", "expected_parent_ids": {229}, "expected_answer": "$319 million"},
    {"question": "How much did CBA spend on technology investment during the half?", "expected_parent_ids": {166}, "expected_answer": "$1,207 million"},
    {"question": "How much did CBA lend to businesses during the half?", "expected_parent_ids": {180}, "expected_answer": "$25 billion"},
    {"question": "How much capital did CBA return to shareholders during the half?", "expected_parent_ids": {183}, "expected_answer": "$4.4 billion"},
    {"question": "How many homes did CBA help customers buy during the half?", "expected_parent_ids": {173}, "expected_answer": "More than 79,000 homes"},
    {"question": "How many regional branches does CBA operate?", "expected_parent_ids": {206}, "expected_answer": "281 regional branches"},
    {"question": "What was CBA's dividend payout ratio for the half?", "expected_parent_ids": {155}, "expected_answer": "Approximately 74% of cash NPAT"},
    {"question": "How much value has CommBank Yello delivered to retail customers?", "expected_parent_ids": {208}, "expected_answer": "More than $190 million"},
]
