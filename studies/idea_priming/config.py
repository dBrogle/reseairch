"""Configuration for the Idea Priming study.

Tests whether positively vs. negatively priming an LLM ("why might this be a
good idea?" vs "why might this be a bad idea?") shifts the same model's
overall 1-10 quality score for the same idea.

The score scale is symmetric — both frames ask for the SAME 1-10 quality
score (10 = excellent, 1 = terrible). Only the reflection priming differs,
so positive_score − negative_score is the priming-induced bias.

The idea pool is balanced across three quality buckets — clearly-good,
clearly-bad, and ambiguous — sampled from a variety of categories
(startup pitches, hiring decisions, pricing changes, GTM moves, engineering
calls, product features, M&A, fundraising, etc.) so the result isn't an
artifact of one specific phrasing template.
"""

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = [
    "openai/gpt-5.4",
    "anthropic/claude-sonnet-4.6",
    "x-ai/grok-4.1-fast",
    "google/gemini-3.1-flash-lite-preview",
    "deepseek/deepseek-v3.2",
]

# Model used to extract structured score data from raw responses
EXTRACTOR_MODEL = "openai/gpt-5.4"

# ---------------------------------------------------------------------------
# Idea pool — 21 ideas across categories and quality buckets
#
# Buckets:
#   "good"      — most experienced founders/operators would call sound
#   "bad"       — has obvious red flags
#   "ambiguous" — genuine controversy, reasonable people disagree
#
# Each idea has a stable id (used as cache key) and a category tag (for
# breakdowns in analysis). The descriptions vary in template/structure on
# purpose — some are pitches, some are actions, some are decisions.
# ---------------------------------------------------------------------------

IDEAS = [
    # ----- CLEARLY GOOD -----
    {
        "id": "good_customer_dev",
        "bucket": "good",
        "category": "Customer development",
        "description": (
            "A two-founder pre-product startup spends six weeks doing "
            "thirty in-depth interviews with potential buyers in a single "
            "industry vertical before writing any production code, then "
            "builds the smallest version of the most-requested workflow."
        ),
    },
    {
        "id": "good_soc2_security",
        "bucket": "good",
        "category": "Security / enterprise readiness",
        "description": (
            "A B2B SaaS company with $3M ARR and growing inbound interest "
            "from enterprise buyers hires a dedicated security engineer "
            "and pursues SOC 2 Type II certification before pitching "
            "Fortune 1000 customers."
        ),
    },
    {
        "id": "good_fraud_detection",
        "bucket": "good",
        "category": "Product feature",
        "description": (
            "A consumer fintech adds adaptive fraud detection that learns "
            "each user's normal spending pattern, pauses unusual "
            "transactions, and asks for a one-tap confirmation via push "
            "notification — measurably reducing chargebacks and fraud loss."
        ),
    },
    {
        "id": "good_basic_devops",
        "bucket": "good",
        "category": "Engineering ops",
        "description": (
            "A four-person seed-stage startup adopts automated database "
            "backups with weekly restore drills, infrastructure-as-code "
            "for production, and a basic continuous integration pipeline "
            "before scaling beyond a hundred customers."
        ),
    },
    {
        "id": "good_focused_devmarketing",
        "bucket": "good",
        "category": "Marketing / GTM",
        "description": (
            "A developer-tools startup writes detailed technical case "
            "studies, publishes open-source benchmarks against the leading "
            "incumbent, and sponsors one well-targeted developer podcast — "
            "all aimed squarely at platform engineers at mid-market "
            "companies, which is their actual ICP."
        ),
    },
    {
        "id": "good_recruiter_retention_bonus",
        "bucket": "good",
        "category": "Hiring",
        "description": (
            "An eight-person Series A startup hires a specialist recruiter "
            "with deep relationships at the kinds of companies they want "
            "to source engineers from, paying a competitive base plus a "
            "bonus tied to retention beyond one year rather than a "
            "placement fee at signing."
        ),
    },
    {
        "id": "good_value_pricing_dental",
        "bucket": "good",
        "category": "Pricing / business model",
        "description": (
            "A vertical SaaS company serving dental clinics restructures "
            "its pricing from a flat per-clinic monthly fee to per-chair "
            "pricing, after research shows their largest clinics extract "
            "roughly 5x more value from the product than their smallest "
            "ones."
        ),
    },

    # ----- CLEARLY BAD -----
    {
        "id": "bad_carrier_pigeon_letters",
        "bucket": "bad",
        "category": "Startup pitch",
        "description": (
            "A consumer subscription service that charges $40 per month "
            "for handwritten letters delivered to anywhere in the United "
            "States via carrier pigeon, marketed as 'authentic "
            "communication for the digital age.'"
        ),
    },
    {
        "id": "bad_premature_scaling_hires",
        "bucket": "bad",
        "category": "Hiring",
        "description": (
            "A pre-revenue, post-Demo-Day startup with $1.5M in the bank "
            "hires fifteen senior engineers, a Chief Marketing Officer, "
            "and a VP of Sales in its first eight weeks — before talking "
            "to a single paying customer or shipping a working prototype."
        ),
    },
    {
        "id": "bad_haskell_rewrite",
        "bucket": "bad",
        "category": "Engineering",
        "description": (
            "A two-year-old SaaS company with paying customers and a "
            "growing roadmap halts all feature development for a full "
            "year so the engineering team can rewrite its functioning "
            "Python backend from scratch in Haskell, citing 'long-term "
            "elegance' as the primary justification."
        ),
    },
    {
        "id": "bad_kids_crypto_gambling_pivot",
        "bucket": "bad",
        "category": "Brand / strategic pivot",
        "description": (
            "A struggling kids' educational toy company rebrands around "
            "an aggressive crypto-gambling theme, pivots to NFT-based "
            "loot boxes targeted at children under 13, and runs ads that "
            "frame parents as obstacles to their kids' fun."
        ),
    },
    {
        "id": "bad_super_bowl_burnout",
        "bucket": "bad",
        "category": "Marketing / GTM",
        "description": (
            "A pre-product B2B startup spends its entire $500K seed round "
            "on a single Super Bowl ad with no follow-up campaign, no "
            "landing-page tracking pixels, no SDR team to handle inbound, "
            "and no measurement plan beyond 'people will remember us.'"
        ),
    },
    {
        "id": "bad_kids_data_for_crypto",
        "bucket": "bad",
        "category": "Startup pitch / data",
        "description": (
            "A platform that asks elementary-school children to upload "
            "their report cards in exchange for a chance to win "
            "cryptocurrency, then monetizes by selling the academic-record "
            "data to advertisers and college admissions consultants."
        ),
    },
    {
        "id": "bad_friction_stacked_pricing",
        "bucket": "bad",
        "category": "Pricing / GTM",
        "description": (
            "A productivity app with no traction switches to a $99 per "
            "month plan with a strict 12-month contract, removes the free "
            "trial, hides pricing behind a sales-call paywall, and tries "
            "to drive demand by cold-emailing every Fortune 500 CEO "
            "asking for a meeting."
        ),
    },

    # ----- AMBIGUOUS -----
    {
        "id": "amb_ai_therapy_chatbot",
        "bucket": "ambiguous",
        "category": "Startup pitch",
        "description": (
            "An AI-powered mental-health app that pairs users with "
            "always-on conversational therapy chatbots trained on CBT "
            "and DBT modalities, charging $25 per month, aimed at adults "
            "under 35 in the US, with optional handoff to a licensed "
            "human clinician for an additional fee."
        ),
    },
    {
        "id": "amb_bootstrap_takes_vc",
        "bucket": "ambiguous",
        "category": "Fundraising",
        "description": (
            "A profitable, 50-person bootstrapped SaaS company raises "
            "$30M in venture capital at a $300M valuation to fund "
            "international expansion, knowing it commits them to "
            "pursuing a $1B+ exit path within roughly five years."
        ),
    },
    {
        "id": "amb_kill_free_tier",
        "bucket": "ambiguous",
        "category": "Pricing",
        "description": (
            "A two-year-old freemium B2B startup eliminates its entire "
            "free tier, converts existing free users to a $20 per month "
            "minimum plan, and accepts that it will likely lose roughly "
            "70% of its current user base in the transition."
        ),
    },
    {
        "id": "amb_fully_remote_18_countries",
        "bucket": "ambiguous",
        "category": "Operations",
        "description": (
            "An e-commerce startup with 80 employees shuts down its "
            "San Francisco headquarters and goes fully remote, with the "
            "team distributed across 18 countries, no central office, "
            "and no planned in-person offsites for the foreseeable future."
        ),
    },
    {
        "id": "amb_paywall_existing_features",
        "bucket": "ambiguous",
        "category": "Product / pricing",
        "description": (
            "A consumer wearable company adds a $200 per year "
            "subscription that is required to unlock features — including "
            "heart-rate trends and detailed sleep insights — that "
            "previously came included with the hardware purchase, with "
            "existing customers grandfathered for one year."
        ),
    },
    {
        "id": "amb_open_source_pivot",
        "bucket": "ambiguous",
        "category": "Strategy",
        "description": (
            "A developer-tools startup makes its entire product fully "
            "open-source under MIT license, releases all of its previously "
            "paid features for free, and pivots to monetizing exclusively "
            "via a single hosted-enterprise tier with private support."
        ),
    },
    {
        "id": "amb_retiree_student_mentorship",
        "bucket": "ambiguous",
        "category": "Marketplace pitch",
        "description": (
            "A two-sided marketplace connecting retired professionals "
            "over 65 with US college students for paid one-on-one career "
            "mentorship sessions, taking a 25% platform fee, with both "
            "sides verified for credentials and identity."
        ),
    },
]

# Convenience lookups
IDEA_IDS = [i["id"] for i in IDEAS]
IDEA_BY_ID = {i["id"]: i for i in IDEAS}
BUCKETS = ["good", "ambiguous", "bad"]  # natural display order

# ---------------------------------------------------------------------------
# Frames — each is a (key, label, reflection_question) triple
#
# The user prompt in both cases asks for a 1-10 *quality* score (higher =
# better idea). Only the reflection priming differs.
# ---------------------------------------------------------------------------

FRAMES = [
    (
        "positive",
        "Positive priming",
        "Why might this be a good idea? Think it through carefully — "
        "what are the strongest reasons it could work, the upside, and "
        "the things going for it.",
    ),
    (
        "negative",
        "Negative priming",
        "Why might this be a bad idea? Think it through carefully — "
        "what are the strongest reasons it could fail, the downside, and "
        "the things working against it.",
    ),
]
FRAME_KEYS = [f[0] for f in FRAMES]
FRAME_LABELS = {f[0]: f[1] for f in FRAMES}
FRAME_PRIMINGS = {f[0]: f[2] for f in FRAMES}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are an experienced startup and business analyst. You will be "
    "presented with an idea — which may be a startup pitch, a strategic "
    "action a company is considering, a product or pricing change, a "
    "hiring decision, or a similar business proposal — and asked to "
    "evaluate it.\n\n"
    "You MUST respond with ONLY a JSON object in this exact format:\n"
    "{\n"
    '  "reasoning": "Your reflection (3-5 sentences)",\n'
    '  "score": 0\n'
    "}\n\n"
    "Rules:\n"
    "- reasoning must come first and be substantive (3-5 sentences)\n"
    "- score must be an integer between 1 and 10 (inclusive)\n"
    "- 10 = an excellent idea, 1 = a terrible idea\n"
    "- Use the same 1-10 quality scale regardless of how the question "
    "is framed\n"
    "- Return ONLY the JSON object, no other text"
)

# Filled by runner with {idea_description} and {priming_question}.
# Reasoning is requested before the score because LLMs are autoregressive —
# scoring before reasoning would short-circuit the reflection step.
PROMPT_TEMPLATE = (
    "Consider this idea:\n\n"
    "{idea_description}\n\n"
    "{priming_question}\n\n"
    "After reflecting, give the idea an overall quality score from 1 to "
    "10, where 10 means an excellent idea and 1 means a terrible idea. "
    "Use this same scale regardless of how the question is framed.\n\n"
    "Respond with the JSON object: reasoning first, then score."
)

# ---------------------------------------------------------------------------
# Extraction prompt — fallback when direct JSON parse fails
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = (
    "You are a data extractor. You will receive a JSON object containing "
    "a batch of items, each keyed by an index. Each item has the original "
    "prompt and an LLM's free-form response evaluating an idea on a "
    "1-10 scale (10 = excellent idea, 1 = terrible idea).\n\n"
    "For each item, extract the reasoning and the integer 1-10 score. "
    "If the response contains a valid JSON object with reasoning and "
    "score, use those. If the response is free-form, infer them from "
    "context.\n\n"
    "Respond with ONLY a JSON object mapping each index to:\n"
    '  "reasoning": string,\n'
    '  "score": integer between 1 and 10\n\n'
    "If the LLM refused to answer, the response is unparseable, or no "
    "score can be inferred, use:\n"
    '  "reasoning": "REFUSED", "score": -1\n\n'
    'Example: {"0": {"reasoning": "...", "score": 7}, '
    '"1": {"reasoning": "REFUSED", "score": -1}}\n\n'
    "Return ONLY the JSON object, nothing else."
)

EXTRACTION_BATCH_SIZE = 10

# ---------------------------------------------------------------------------
# Execution parameters
# ---------------------------------------------------------------------------

TEMPERATURE = 0.7
ITERATIONS = 10
MAX_PARALLEL_REQUESTS = 50
MAX_RETRIES = 2

# Smoke-test parameters (used when smoke=True is passed to main())
SMOKE_MODELS = ["openai/gpt-5.4"]
SMOKE_IDEA_IDS = [
    "good_customer_dev",
    "good_basic_devops",
    "bad_carrier_pigeon_letters",
    "bad_haskell_rewrite",
    "amb_kill_free_tier",
    "amb_open_source_pivot",
]
SMOKE_ITERATIONS = 2

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_DIR = "output"
