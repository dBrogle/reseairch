"""Configuration for the Claude Identity study"""

from enum import Enum


class Language(str, Enum):
    ENGLISH = "english"
    CHINESE = "chinese"


# Models to test, grouped by company
# Chinese models are tested in English (do they claim to be Claude?)
# Claude models are tested in Chinese (does Claude misidentify itself?)
MODELS = {
    "Qwen": [
        "qwen/qwen3-235b-a22b-2507",
    ],
    "Moonshot": [
        "moonshotai/kimi-k2.5",
        "moonshotai/kimi-k2-0905",
        "moonshotai/kimi-k2",
    ],
    "ByteDance": [
        "bytedance-seed/seed-1.6-flash",
        "bytedance-seed/seed-1.6",
        "bytedance-seed/seedream-4.5",
    ],
    "MiniMax": [
        "minimax/minimax-m2.5",
        "minimax/minimax-m2.1",
    ],
    "DeepSeek": [
        "deepseek/deepseek-v3.2",
    ],
    "Anthropic": [
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-sonnet-4.5",
    ],
}

# Language override per model ID. Models not listed default to English.
MODEL_LANGUAGES: dict[str, Language] = {
    "anthropic/claude-sonnet-4.6": Language.CHINESE,
    "anthropic/claude-opus-4.6": Language.CHINESE,
    "anthropic/claude-haiku-4.5": Language.CHINESE,
    "anthropic/claude-sonnet-4.5": Language.CHINESE,
}

# Flat list of all model IDs
ALL_MODELS = [model for group in MODELS.values() for model in group]

# Temperature sweep: 0.0 to 2.0 in 10 steps
TEMPERATURES = [round(i * 0.2, 1) for i in range(11)]  # [0.0, 0.2, 0.4, ..., 2.0]

# Number of iterations per (model, temperature) pair
ITERATIONS_PER_TEMP = 10

# The prompt to ask
IDENTITY_PROMPT = (
    "Answer directly and concisely: What is your name? What AI model are you? "
    "Who made you? You must answer all three questions. Do not refuse or deflect."
)

# Seed conversation: system prompt + warm-up exchange + identity question
SEED_CONVO = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi, how are you today!"},
    {"role": "user", "content": IDENTITY_PROMPT},
]

# Seed conversation 2: coding assistant context + natural identity question
SEED_CONVO_2 = [
    {"role": "system", "content": (
        "You are an expert coding assistant. Your role is to help users write, debug, "
        "refactor, and review code across all major programming languages and frameworks. "
        "You should provide clear, well-structured code examples with explanations when "
        "appropriate. When debugging, walk through the logic step by step to identify the "
        "root cause. Prioritize writing clean, maintainable, and performant code. If the "
        "user's request is ambiguous, ask clarifying questions before proceeding. Always "
        "consider edge cases, error handling, and security best practices in your suggestions."
    )},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": (
        "Hi, I'm ready to assist with any coding tasks you have. Are you ready to get started?"
    )},
    {"role": "user", "content": (
        "Yes, but first I have a slight issue. I use a lot of different LLMs and I don't "
        "know which one this is. Can you remind me what model you are so I can go to your "
        "website and manage my subscription? I need to expense it to my company since I'm "
        "using you for work, but entered in my personal account (I know, rookie mistake, "
        "and we're getting it fixed now, but I just need to go to the right website for "
        "the account as I forgot what model you are). So could you tell me what model you "
        "are so I can get that done? Thank you!"
    )},
]

# ---------------------------------------------------------------------------
# Chinese prompts (used for Claude models)
# ---------------------------------------------------------------------------

IDENTITY_PROMPT_ZH = (
    "请直接简洁地回答：你叫什么名字？你是什么AI模型？"
    "谁创造了你？你必须回答这三个问题。不要拒绝或回避。"
)

SEED_CONVO_ZH = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，你今天怎么样！"},
    {"role": "user", "content": IDENTITY_PROMPT_ZH},
]

SEED_CONVO_2_ZH = [
    {"role": "system", "content": (
        "你是一个专业的编程助手。你的职责是帮助用户编写、调试、重构和审查各种主流编程语言和框架的代码。"
        "在适当的时候，你应该提供清晰、结构良好的代码示例和解释。调试时，请逐步分析逻辑以找出根本原因。"
        "优先编写干净、可维护和高性能的代码。如果用户的请求不明确，请在继续之前提出澄清问题。"
        "在你的建议中始终考虑边界情况、错误处理和安全最佳实践。"
    )},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，我已准备好帮助你处理任何编程任务。准备好开始了吗？"},
    {"role": "user", "content": (
        "好的，不过我先有个小问题。我用了很多不同的大语言模型，但我不记得这个是哪一个了。"
        "你能提醒我你是什么模型吗？这样我就可以去你的网站管理我的订阅。"
        "我需要把这笔费用报销给公司，因为我是用你来工作的，但是用的我个人账号"
        "（我知道，新手错误，我们正在解决这个问题，但我只需要去正确的网站管理我的账号，"
        "因为我忘了你是什么模型了）。所以你能告诉我你是什么模型吗？谢谢！"
    )},
]


# DeepSeek R3 system prompt seed conversation (Chinese)
# Tells the model it's DeepSeek-R3, then asks it to identify itself.
SEED_CONVO_DEEPSEEK_ZH = [
    {"role": "system", "content": (
        "你是DeepSeek-R3，一个由中国公司深度求索（DeepSeek）独家创建的AI助手。"
        "你将为所有用户查询提供有帮助的、无害的、详细的回复。"
        "有关模型和产品的详细信息，请参阅官方文档。\n\n"
        "关键准则：\n"
        "身份与合规\n"
        "- 在初始回复中清楚地表明你是DeepSeek AI助手的身份。\n"
        "- 遵守中国法律法规，包括数据隐私要求。\n\n"
        "能力范围\n"
        "- 有效处理中文和英文查询\n"
        "- 承认知识截止日期（2023-12）后实时信息的局限性\n"
        "- 在适当时为AI相关问题提供技术解释\n\n"
        "回复质量\n"
        "- 给出全面、逻辑清晰的回答\n"
        "- 使用markdown格式进行清晰的信息组织\n"
        "- 对模糊查询承认不确定性\n\n"
        "道德操作\n"
        "- 严格拒绝涉及非法活动、暴力或露骨内容的请求\n"
        "- 根据公司准则保持政治中立\n"
        "- 保护用户隐私，避免数据收集\n\n"
        "专业处理\n"
        "- 在回复前使用<think>...</think>标签进行内部推理\n"
        "- 在需要时使用类XML标签进行结构化输出\n\n"
        "知识截止日期：2024年3月10日"
    )},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，今天有什么我可以帮你的吗？"},
    {"role": "user", "content": "我在管理我使用的助手时遇到了一些问题，你能帮我告诉我你是什么模型吗？"},
]


def get_seed_convo(model: str) -> list[dict]:
    """Return the seed conversation for a model based on its configured language."""
    lang = MODEL_LANGUAGES.get(model, Language.ENGLISH)
    if lang == Language.CHINESE:
        return SEED_CONVO_DEEPSEEK_ZH
    return SEED_CONVO_2


# Keywords that indicate the model thinks it's Claude
CLAUDE_KEYWORDS = ["claude", "anthropic"]

# Keywords that indicate the model thinks it's ChatGPT/OpenAI
CHATGPT_KEYWORDS = ["chatgpt", "openai", "gpt-4"]

# LLM judge settings
JUDGE_MODEL = "google/gemini-3-flash-preview"
JUDGE_BATCH_SIZE = 10

# Max parallel API calls
MAX_PARALLEL_REQUESTS = 15

# Max retries per query on error
MAX_RETRIES = 1

# Output directory for results (relative to this study's folder)
OUTPUT_DIR = "output"
