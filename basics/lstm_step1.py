import math

f = 0.9
i = 1.0
o = 1.0

c = 0.0
h = 0.0
sentence = [1.0, 0, 0, 0, 0]

for t,x in enumerate(sentence):
    new = math.tanh(x)
    c = f * c + i * new
    h = o * math.tanh(c)
    print(f"t={t}, x={x}, c={c:.4f}, h={h:.4f}")                    