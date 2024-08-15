"""NOTE: The script is not related to opencv, I just wanted to implement mandelbrot and julia sets"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# the most abstracted method
# Image.effect_mandelbrot((4000, 4000), (-3, -2.5, 2, 2.5), 100).show()


# function to yield both mandelbrot and julia sets
def sequence(c, z=0):
    while True:
        yield z
        z = z**2 + c


# showcase of difference for a number to be in mandelbrot and julia sets
def mandelbrot(candidate):
    return sequence(c=candidate, z=0)


def julia(candidate, parameter):
    return sequence(c=parameter, z=candidate)


mand_gen = mandelbrot(1)
julia_gen = julia(1, 0.25)

for _ in range(10):
    print(next(mand_gen))
for _ in range(10):
    print(next(julia_gen))
