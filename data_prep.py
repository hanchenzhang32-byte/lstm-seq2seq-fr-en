"""第 1 课：数据预处理

目标：fra.txt  →  训练 8000 / 验证 1000 / 测试 1000 个 (法语 tokens, 英语 tokens) 句对

流程：
    原始一行  "Go.\tVa !\tCC-BY 2.0 ..."
      → split('\t')           拿到 en, fr（第三列来源信息不用）
      → normalize()           统一特殊空白、引号，转小写
      → tokenize()            切成 token 列表
      → 过滤                  两边非空 且 各 ≤ 10 个 token
      → 按法语去重            同一句法语只保留第一次出现的那条
      → Random(42).shuffle    打乱
      → 取前 10000 并切分     8000 / 1000 / 1000

运行  .venv/bin/python check_lesson1.py  自检。
"""

import random
import re
import unicodedata

DATA_PATH = "data/fra.txt"
MAX_LEN = 10          # 每种语言最多 10 个 token（标点算，特殊 token 不算）
NUM_PAIRS = 10_000
SPLIT_SIZES = (8_000, 1_000, 1_000)
SEED = 42

# 观察数据时发现的特殊字符。法语排版会在 ! ? : 前放“窄不换行空格”，
# 这些字符肉眼看和空格一样，但 Python 里是不同的字符，必须先统一掉。
SPECIAL_CHARS = {
    "\xa0": " ",      # NO-BREAK SPACE
    " ": " ",    # NARROW NO-BREAK SPACE
    " ": " ",    # THIN SPACE
    "​": "",     # ZERO WIDTH SPACE（直接删掉）
    "\xad": "",       # SOFT HYPHEN（直接删掉）
    "’": "'",    # ’ 弯引号 → 直引号
    "‘": "'",    # ‘
}

# 分词正则：
#   \w+      一个或多个“单词字符”（字母、数字、下划线；Python 3 中包含 é ç à 等）
#   |        或者
#   [^\w\s]  恰好一个“既不是单词字符、也不是空白”的字符，即一个标点
# re.compile 把正则预先编译好，之后反复调用更快。
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def normalize(s: str) -> str:
    """规范化一个句子：NFC → 替换特殊字符 → 小写 → 去首尾空白。

    例：normalize("Je n’ai pas peur !") == "je n'ai pas peur !"
    """
    s = unicodedata.normalize("NFC", s)       # "e" + "◌́" 两个字符 → "é" 一个字符
    for old, new in SPECIAL_CHARS.items():    # 逐个替换特殊空白和弯引号
        s = s.replace(old, new)
    return s.lower().strip()                  # 小写：Go 和 go 算同一个词


def tokenize(s: str) -> list[str]:
    """先 normalize，再按正则切分。

    例：tokenize("Don't go!") == ["don", "'", "t", "go", "!"]
    """
    # findall 从左到右扫描，返回所有匹配片段组成的列表；空白不匹配任何分支，自然被丢弃
    return TOKEN_RE.findall(normalize(s))


def load_pairs(path: str = DATA_PATH) -> list[tuple[list[str], list[str]]]:
    """读文件 → 分词 → 过滤 → 按法语去重。返回 [(fr_tokens, en_tokens), ...]，保持原文件顺序。"""
    pairs = []        # 结果列表，按原文件顺序追加
    seen = set()      # 已经出现过的法语句子；set 的 in 查询是 O(1)，列表是 O(n)

    with open(path, encoding="utf-8") as f:   # with 保证读完自动关闭文件
        for line in f:                        # 逐行读，不会一次把 36MB 全读进内存
            # 每行末尾带 "\n"，先去掉再按 TAB 切成三列
            en, fr, _source = line.rstrip("\n").split("\t")
            fr_tokens = tokenize(fr)          # 第二列：法语 → 模型输入
            en_tokens = tokenize(en)          # 第一列：英语 → 模型目标

            # ① 过滤：任意一边为空或超过 MAX_LEN 就跳过
            if not fr_tokens or not en_tokens:
                continue
            if len(fr_tokens) > MAX_LEN or len(en_tokens) > MAX_LEN:
                continue

            # ② 去重：必须在过滤之后，否则不合格的行会“占掉”这句法语的名额
            key = " ".join(fr_tokens)         # 列表不能放进 set（不可哈希），转成字符串
            if key in seen:
                continue
            seen.add(key)

            pairs.append((fr_tokens, en_tokens))

    return pairs


def split_pairs(pairs):
    """用独立的 Random(SEED) 打乱，取前 NUM_PAIRS 条，依次切成 train / val / test。"""
    pairs = list(pairs)                        # 复制一份，shuffle 是原地操作，不能改动调用者的列表
    random.Random(SEED).shuffle(pairs)         # 独立随机数生成器：不受其他代码调用 random 的影响
    pairs = pairs[:NUM_PAIRS]

    n_train, n_val, _ = SPLIT_SIZES
    train = pairs[:n_train]                    # [0, 8000)
    val = pairs[n_train:n_train + n_val]       # [8000, 9000)
    test = pairs[n_train + n_val:]             # [9000, 10000)
    return train, val, test


if __name__ == "__main__":
    pairs = load_pairs()
    train, val, test = split_pairs(pairs)
    print(f"合格句对 {len(pairs)}，train/val/test = {len(train)}/{len(val)}/{len(test)}")
    for fr, en in train[:5]:
        print(" ".join(fr), "  =>  ", " ".join(en))
