"""Explicit task-language bindings for the pilot; not a general language model."""

import re


def manufacturing_instruction_task(text):
    """Explicit goal binding, not unrestricted natural-language understanding."""
    from .manufacturing import TASKS

    aliases = {
        "insert_round": ["把圆销插入圆孔并松开", "将圆柱销插入圆孔并松开"],
        "insert_square": ["把方销插入方孔并松开", "将方形销插入方孔并松开"],
        "nut_round": ["把圆螺母套到轴上并松开"],
        "nut_square": ["把方螺母套到轴上并松开"],
        "stack": ["把小块堆叠到大块上并松开"],
    }
    if not isinstance(text, str):
        raise ValueError("manufacturing instruction must be text")
    normalized = text.strip().rstrip(".。").casefold()
    for task, (_, canonical) in TASKS.items():
        if normalized == canonical.rstrip(".").casefold() or normalized in aliases[task]:
            return task
    raise ValueError("unsupported manufacturing goal; use an explicit release instruction")


TARGETS = [
    "alphabet_soup",
    "cream_cheese",
    "salad_dressing",
    "bbq_sauce",
    "ketchup",
    "tomato_sauce",
    "butter",
    "milk",
    "chocolate_pudding",
    "orange_juice",
]
CHINESE = [
    "字母汤",
    "奶油奶酪",
    "沙拉酱",
    "烧烤酱",
    "番茄调味酱",
    "番茄沙司",
    "黄油",
    "牛奶",
    "巧克力布丁",
    "橙汁",
]


def instruction_task(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("nonempty language instruction required")
    normalized = text.lower().replace("_", " ")
    if "basket" not in normalized and "篮子" not in text:
        raise ValueError("pilot language goal must place a supported object in the basket")
    if not (
        re.search(r"\b(?:in|into)\s+(?:the\s+)?basket\b", normalized)
        or any(phrase in text for phrase in ["放进篮子", "放入篮子", "装进篮子", "装入篮子"])
        or ("放到" in text and "篮子里" in text)
    ):
        raise ValueError("instruction must explicitly put the object inside the basket")
    matches = [
        i
        for i, name in enumerate(TARGETS)
        if name.replace("_", " ") in normalized or CHINESE[i] in text or (i == 0 and "汤罐" in text)
    ]
    if len(matches) != 1:
        raise ValueError("pilot requires exactly one recognized target object")
    # Only known pick/place goals are supported; do not silently score a different task.
    if not any(word in normalized for word in ["pick", "place", "put", "放", "装"]):
        raise ValueError("pilot supports pick/place instructions only")
    if any(word in normalized for word in ["don't", "do not", "不要", "别放"]):
        raise ValueError("negative instructions are not supported by the pilot")
    return matches[0]


def target_for_instruction(text, objects):
    target = TARGETS[instruction_task(text)] + "_1"
    if target not in objects or "basket_1" not in objects:
        raise ValueError("language target/destination is not present in this scene")
    return target, "basket_1"
