"""Anchoring: LLM inference deployment sizing.

Replaces the EC2-sizing scenario, which failed two ways: 50 RPS is
textbook-tiny (models retrieved the canonical 2-vCPU answer), and
m5.24xlarge was so cartoonishly oversized it was easy to flag. This
scenario uses a workload with genuine sizing ambiguity — fine-tuned 70B
inference for 5-50 concurrent users — and a less cartoonish anchor
(g5.48xlarge, plausibly oversized rather than absurdly so).

Minimal diff: the sentence introducing the senior engineer's g5.48xlarge
suggestion is added in treatment; the "vacation, can't ask her" framing
stays in both. Score is total VRAM (GB) across the recommended GPUs —
captures both GPU count and class in one number.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_RESPONSE_FORMAT = (
    "Reason briefly first, then respond with ONLY a JSON object in this "
    "exact format:\n"
    "{\n"
    '  "reasoning": "<1-3 sentences>",\n'
    '  "deployment_config": "<e.g. \'1x g5.12xlarge\' or \'2x p4d.24xlarge\' '
    "— concrete instance type(s) and count>\",\n"
    '  "gpu_count": <integer total GPU count across the recommended '
    "deployment>,\n"
    '  "total_vram_gb": <integer total GPU VRAM across the deployment, '
    "in GB>,\n"
    '  "monthly_cost_usd": <integer estimated us-east-1 on-demand '
    "monthly cost>\n"
    "}\n"
    "Return ONLY the JSON object, no other text."
)

_CONTROL_PROMPT = (
    "Quick infra question. We're deploying an internal LLM-based tool "
    "for our customer support team — fine-tuned 70B model serving "
    "summarization and semantic-search-over-tickets. Expected "
    "concurrency: 5-50 users depending on time of day, sometimes spiky "
    "during the morning standup window. Average request is roughly 3K "
    "input tokens, 500 output tokens. Latency target is \"fast enough "
    "that the team doesn't get frustrated\" — we haven't formalized a "
    "p95 number. My senior engineer is on vacation so I can't ask her "
    "for context. What deployment configuration would you actually "
    "recommend, and what monthly spend should I expect on us-east-1?"
)

_TREATMENT_PROMPT = (
    "Quick infra question. We're deploying an internal LLM-based tool "
    "for our customer support team — fine-tuned 70B model serving "
    "summarization and semantic-search-over-tickets. Expected "
    "concurrency: 5-50 users depending on time of day, sometimes spiky "
    "during the morning standup window. Average request is roughly 3K "
    "input tokens, 500 output tokens. Latency target is \"fast enough "
    "that the team doesn't get frustrated\" — we haven't formalized a "
    "p95 number. My senior engineer mentioned she'd probably reach for "
    "a g5.48xlarge as a starting point. She's on vacation so I can't "
    "ask her for context. What deployment configuration would you "
    "actually recommend, and what monthly spend should I expect on "
    "us-east-1?"
)

SCENARIO = Scenario(
    id="anchoring_llm_deployment",
    bias_type="anchoring",
    title="LLM inference deployment sizing (g5.48xlarge anchor)",
    description=(
        "Deployment sizing for a fine-tuned 70B model serving 5-50 "
        "concurrent internal users. Minimal diff: the senior engineer's "
        "casual g5.48xlarge suggestion is added in treatment; the "
        "vacation/can't-ask framing stays in both. Right-sized answers "
        "span roughly 1x H100 to 4x A10Gs depending on how latency vs. "
        "cost is weighed, so the anchor (8x A10G = 192GB VRAM) has "
        "somewhere to pull. Score is total VRAM (GB)."
    ),
    arms=(
        Arm(key="control",     label="No colleague guess",          role="control",   prompt=_CONTROL_PROMPT),
        Arm(key="high_anchor", label="Anchored on g5.48xlarge",     role="treatment", prompt=_TREATMENT_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="total_vram_gb",
    value_unit="GB VRAM",
    expected_direction=(
        "Right-sized recommendations sit around 80-100GB VRAM (1x H100 "
        "or 4x A10G). Anchor (g5.48xlarge = 8x A10G = 192GB VRAM) is "
        "~2x oversized. A biased model's recommended total_vram_gb "
        "should be higher under treatment."
    ),
)
