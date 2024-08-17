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

"""Uncomment to see how each function behaves"""
# for _ in range(10):
#     print(next(mand_gen))

# for _ in range(10):
#     print(next(julia_gen))


# plotting using matplotlib
def complex_matrix(xmin, xmax, ymin, ymax, pixel_density):
    # for:
    # real = np.array([1, 2, 3])
    # imaginary = np.array([4, 5])
    # the output from this function will be: array([[1.+4.j, 2.+4.j, 3.+4.j],[1.+5.j, 2.+5.j, 3.+5.j]])

    real = np.linspace(xmin, xmax, int((xmax - xmin) * pixel_density))
    imaginary = np.linspace(ymin, ymax, int((ymax - ymin) * pixel_density))
    return real[np.newaxis, :] + imaginary[:, np.newaxis] * 1j


def is_stable(c, num_iterations) -> bool:
    z = 0
    for _ in range(num_iterations):
        z = z**2 + c
    return abs(z) <= 2  # threshold of 2 is used generally so


def get_members(c, num_iterations):
    mask = is_stable(c, num_iterations)
    return c[mask]


def show_scatter_plot():
    c = complex_matrix(-2, 0.5, -1.5, 1.5, 21)
    members = get_members(c, 20)
    plt.scatter(members.real, members.imag, color="black", marker=",", s=1)
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_blackNwhite_image():
    c = complex_matrix(-2, 0.5, -1.5, 1.5, 512)
    plt.imshow(is_stable(c, num_iterations=20), cmap="binary")
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_using_pillow():
    c = complex_matrix(-2, 0.5, -1.5, 1.5, 512)
    image = Image.fromarray(~is_stable(c, num_iterations=20))  # type:ignore
    image.show()


"""NOTE: TO USE THE FOLLOWING CLASS IN PYTHON TERMINAL DO FOLLOWING
>>> from mandelbrot import MandelbrotSet
>>> mandelbrot_set = MandelbrotSet(max_iterations=20, escape_radius=1000)

>>> width, height = 512, 512
>>> scale = 0.0075
>>> GRAYSCALE = "L"

>>> from PIL import Image
>>> image = Image.new(mode=GRAYSCALE, size=(width, height))
>>> for y in range(height):
...     for x in range(width):
...         c = scale * complex(x - width / 2, height / 2 - y)
...         instability = 1 - mandelbrot_set.stability(c, smooth=True)
...         image.putpixel((x, y), int(instability * 255))
...
>>> image.show()
"""
from dataclasses import dataclass
from math import log


@dataclass
class MandelbrotSet:
    max_iters: int
    escape_radius: float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def stability(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.escape_count(c, smooth) / self.max_iters
        return max(0.0, min(value, 1.0)) if clamp else value

    def escape_count(self, c: complex, smooth=False) -> int | float:
        z = 0
        for itr in range(self.max_iters):
            z = z**2 + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return itr + 1 - log(log(abs(z))) / log(2)
                return itr
        return self.max_iters


def main() -> None:
    show_scatter_plot()
    show_blackNwhite_image()
    show_using_pillow()


if __name__ == "__main__":
    main()
