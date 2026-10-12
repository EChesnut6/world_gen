from opensimplex import OpenSimplex   # pip install opensimplex
x = 5
y = 5
n = OpenSimplex(seed=1).noise2(x, y)
print(n)

def fbm(noise_fn, octaves=5, lacunarity=2.0, gain=0.5):
    total, amp = 0, 1
    for o in range(octaves):
        total += amp * noise_fn(cell=int(128 / lacunarity**o) or 1)
        amp *= gain
    return total
# turbulence: sum abs(noise); ridged: 1 - abs(noise)