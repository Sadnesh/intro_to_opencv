"""NOTE: The script is not related to opencv, I just wanted to implement mandelbrot and julia sets"""

import numpy as np
from PIL import Image, ImageEnhance
from PIL.ImageColor import getrgb
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d

black = (0, 0, 0)
blue = (0, 0, 1)
maroon = (0.5, 0, 0)
navy = (0, 0, 0.5)
red = (1, 0, 0)
green = (0, 1, 0)
lime = (0, 0.5, 0)
white = (1, 1, 1)

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


@dataclass
class Viewport:
    image: Image.Image
    center: complex
    width: float

    @property
    def height(self):
        return self.scale * self.image.height

    @property
    def offset(self):
        return self.center + complex(-self.width, self.height) / 2

    @property
    def scale(self):
        return self.width / self.image.width

    def __iter__(self):
        for y in range(self.image.height):
            for x in range(self.image.width):
                yield Pixel(self, x, y)


@dataclass
class Pixel:
    viewport: Viewport
    x: int
    y: int

    @property
    def color(self):
        return self.viewport.image.getpixel((self.x, self.y))

    @color.setter
    def color(self, value):
        self.viewport.image.putpixel((self.x, self.y), value)

    def __complex__(self):
        return complex(self.x, -self.y) * self.viewport.scale + self.viewport.offset


# helper function
def paint(
    mandelbrot_set: MandelbrotSet, viewport: Viewport, palette: list, smooth: bool
):
    for pixel in viewport:
        stability = mandelbrot_set.stability(complex(pixel), smooth)
        index = int(min(stability * len(palette), len(palette) - 1))
        pixel.color = palette[index % len(palette)]


def denormalize(palette):
    return [tuple(int(channel * 255) for channel in color) for color in palette]


# gradient functions
def make_gradient(colors, interpolation="linear"):
    X = [i / (len(colors) - 1) for i in range(len(colors))]
    Y = [[color[i] for color in colors] for i in range(3)]
    channels = [interp1d(X, y, kind=interpolation) for y in Y]
    return lambda x: [np.clip(channel(x), 0, 1) for channel in channels]


def make_gradient_palette(num_colors, gradient):
    return denormalize([gradient(i / num_colors) for i in range(num_colors)])


def hsb(hue_degrees: int, saturation: float, brightness: float):
    return getrgb(
        f"hsv({hue_degrees % 360}," f"{saturation * 100}%," f"{brightness * 100}%)"
    )


# it is not optimized so it takes long time to render the image
def run_own_code(image: Image.Image):
    mandel = MandelbrotSet(max_iters=512, escape_radius=1000)
    misiurewicz_point = -0.7435 + 0.1314j
    # ANY OF THESE CAN BE:``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap()`` or ``pyplot.get_cmap()``
    palette = denormalize(matplotlib.colormaps["twilight"].colors)  # type:ignore

    # play with the center value and width to get new positions and various fractals
    viewport = Viewport(image, center=misiurewicz_point, width=0.002)
    paint(mandel, viewport, palette, smooth=True)
    enhancer = ImageEnhance.Brightness(image)
    enhancer.enhance(1.25).show()


def run_using_self_made_palette(image: Image.Image):
    exterior = [(1, 1, 1)] * 50
    interior = [(1, 1, 1)] * 5
    gray_area = [(1 - i / 44,) * 3 for i in range(45)]
    palette = denormalize(exterior + gray_area + interior)
    mandelbrot_set = MandelbrotSet(max_iters=20, escape_radius=1000)
    viewport = Viewport(image, center=-0.75, width=3.5)
    paint(mandelbrot_set, viewport, palette, smooth=True)
    image.show()


def run_gradient_fractal(colors, image: Image.Image):
    num_colors = 256
    mandel = MandelbrotSet(max_iters=20, escape_radius=1000)
    gradient = make_gradient(colors, interpolation="cubic")
    palette = make_gradient_palette(num_colors, gradient)

    viewport = Viewport(image, center=-0.75, width=3.5)
    paint(mandel, viewport, palette, smooth=True)
    image.show()


def run_hsb_fractal(image: Image.Image):
    mand = MandelbrotSet(max_iters=20, escape_radius=1000)
    for pixel in Viewport(image, center=-0.75, width=3.5):
        stability = mand.stability(complex(pixel), smooth=True)
        pixel.color = (
            (0, 0, 0)
            if stability == 1
            else hsb(
                hue_degrees=int(
                    stability * 360
                ),  # changing the value (360 ie degree to take into consideration) will create more interesting results
                saturation=stability,
                brightness=1,
            )
        )
    image.show()


def main() -> None:
    image = Image.new(mode="RGB", size=(512, 512))
    show_scatter_plot()
    show_blackNwhite_image()
    show_using_pillow()
    run_own_code(image)
    run_using_self_made_palette(image)

    colors = [black, navy, blue, maroon, red, black]
    """Some more testings, to remove banding or say to make image more smooth,
        the iters should be increased"""
    # colors = [black, navy, blue, lime, green, white]
    # colors = [white, navy, blue, lime, green, black]
    run_gradient_fractal(colors, image)
    run_hsb_fractal(image)


if __name__ == "__main__":
    main()
