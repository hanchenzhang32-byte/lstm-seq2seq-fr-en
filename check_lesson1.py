"""第 1 课自检：.venv/bin/python check_lesson1.py"""

from data_prep import load_pairs, normalize, split_pairs, tokenize


def check(name, got, expected):
    ok = got == expected
    print(f"[{'通过' if ok else '失败'}] {name}")
    if not ok:
        print(f"    期望: {expected!r}\n    实际: {got!r}")
    return ok


results = [
    check("normalize 特殊空格和弯引号", normalize("  Je n’ai pas peur !  "), "je n'ai pas peur !"),
    check("normalize NFC 合成重音", normalize("Café"), "café"),
    check("tokenize 英文缩写", tokenize("Don't go!"), ["don", "'", "t", "go", "!"]),
    check("tokenize 法语标点", tokenize("Est-ce « vrai »\xa0?"),
          ["est", "-", "ce", "«", "vrai", "»", "?"]),
    check("tokenize 空串", tokenize("   "), []),
]

pairs = load_pairs()
results.append(check("合格且去重后的句对数", len(pairs), 149414))
results.append(check("第一条合格句对", pairs[0], (["va", "!"], ["go", "."])))
results.append(check("去重：'va !' 只出现一次",
                     sum(fr == ["va", "!"] for fr, _ in pairs), 1))
results.append(check("长度都不超过 10",
                     all(0 < len(f) <= 10 and 0 < len(e) <= 10 for f, e in pairs), True))

train, val, test = split_pairs(pairs)
results.append(check("划分大小", (len(train), len(val), len(test)), (8000, 1000, 1000)))
results.append(check("训练集第一条",
                     (" ".join(train[0][0]), " ".join(train[0][1])),
                     ("merci d ' avoir décoré ma maison !", "thank you for decorating my house .")))
results.append(check("没有打乱原列表", pairs[0], (["va", "!"], ["go", "."])))

print(f"\n{sum(results)}/{len(results)} 项通过")
