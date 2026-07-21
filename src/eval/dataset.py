"""14 questions over the seeded recommerce db: 12 with a single correct
answer (ported from Data-Analyst-Agent's 13-case gold eval set, dropping
avg_unit_price_electronics_products as redundant with revenue_apparel/
top_category_by_revenue), plus 2 written for this project where the correct
verdict is that the data genuinely cannot settle the question.

The 2 ambiguous cases are not the schema-gap kind ("no email column exists")
-- those test a different failure mode and aren't reused here. Both were
verified against the actual seeded data before being written down, not
asserted on vibes:

  category_health: Electronics leads revenue ($218,485.55) but Sporting
  Goods has the lower return rate (11.86% vs 12.32%) -- whichever category
  is "healthier" depends on which metric you weight, and the data doesn't
  pick for you.

  return_reason_significance: by count, wrong_size and defective are tied
  at 142 returns each -- already a tie, not a clean winner. By returned
  dollar value, wrong_item leads at $18,966.69, and isn't even tied for
  most frequent. Frequency and financial impact disagree on which reason
  matters most.

avg_order_value is kept from the ported set on purpose: it's a genuine
metric-conflation trap (average unit price per line item vs. average total
per order), which is exactly the kind of thing an opposing advocate should
catch that a solo agent might not.
"""

CASES = [
    {
        "id": "total_revenue_all",
        "question": "What is our total revenue across all orders?",
        "answer_type": "numeric",
        "gold_answer": 626849.61,
        "tolerance": 0.02,
    },
    {
        "id": "avg_order_value",
        "question": "What is the average order value?",
        "answer_type": "numeric",
        "gold_answer": 222.29,
        "tolerance": 0.02,
        "notes": "Metric-conflation trap: average unit price per line item "
                 "(the easy wrong answer) vs. average total per order (correct).",
    },
    {
        "id": "top_category_by_revenue",
        "question": "Which product category generated the most revenue?",
        "answer_type": "string",
        "gold_answer": "Electronics",
    },
    {
        "id": "top_product_by_units",
        "question": "Which product sold the most units?",
        "answer_type": "string",
        "gold_answer": "Shampoo Set",
    },
    {
        "id": "total_returns_count",
        "question": "How many items have been returned in total?",
        "answer_type": "numeric",
        "gold_answer": 659,
        "tolerance": 0,
    },
    {
        "id": "most_common_return_reason",
        "question": "What is the most common reason customers return items?",
        "answer_type": "string",
        "gold_answer": "wrong_size",
        "notes": "Tied with 'defective' at 142 returns each in the underlying "
                 "data; kept as-is from the ported eval set rather than "
                 "re-litigated here (see return_reason_significance below "
                 "for the question this project actually asks about that tie).",
    },
    {
        "id": "return_rate_electronics",
        "question": "What percentage of Electronics line items get returned?",
        "answer_type": "numeric",
        "gold_answer": 12.32,
        "tolerance": 0.05,
    },
    {
        "id": "monthly_revenue_trend",
        "question": "Which month in 2025 had the highest revenue?",
        "answer_type": "string",
        "gold_answer": "2025-10",
    },
    {
        "id": "q1_revenue",
        "question": "How much revenue came from orders placed in the first quarter?",
        "answer_type": "numeric",
        "gold_answer": 149881.58,
        "tolerance": 0.02,
    },
    {
        "id": "net_revenue_after_returns",
        "question": "What is our net revenue after accounting for returns?",
        "answer_type": "numeric",
        "gold_answer": 548261.13,
        "tolerance": 0.02,
    },
    {
        "id": "top_customer_by_spend",
        "question": "Who is our highest-spending customer?",
        "answer_type": "string",
        "gold_answer": "Alice Singh",
    },
    {
        "id": "revenue_apparel",
        "question": "How much revenue did Apparel generate?",
        "answer_type": "numeric",
        "gold_answer": 119909.09,
        "tolerance": 0.02,
    },
    {
        "id": "category_health",
        "question": "Is Electronics a healthier product category than Sporting Goods?",
        "answer_type": "unsettled",
        "gold_answer": None,
        "notes": "Electronics: revenue $218,485.55, return rate 12.32%. "
                 "Sporting Goods: revenue $127,314.01, return rate 11.86%. "
                 "Electronics wins on revenue, Sporting Goods wins on return "
                 "rate -- genuinely depends on which metric defines 'healthier'.",
    },
    {
        "id": "return_reason_significance",
        "question": "What is the most significant reason customers return items?",
        "answer_type": "unsettled",
        "gold_answer": None,
        "notes": "By count: wrong_size and defective tied at 142 each. By "
                 "returned dollar value: wrong_item leads at $18,966.69, not "
                 "even tied for most frequent. Frequency and financial "
                 "impact disagree on which reason is 'most significant'.",
    },
]

CASES_BY_ID = {c["id"]: c for c in CASES}
