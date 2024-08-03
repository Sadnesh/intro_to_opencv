import cv2
from typing import Optional
import numpy as np


def read_image(image_path: str):
    return cv2.imread(image_path)


# display image
def display_image(window_name: str, image):

    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def capture_from_webcam():
    print("Starting webcam now")
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()

        if not ret:
            break

        cv2.imshow("Webcam", frame)

        # least time is good for this case as more time means waiting more and the video's fps decreases
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


def resize_it(width: float, height: float, image, show: bool = False):

    # gives the image size
    og_y, og_x = image.shape[:2]

    x_scale = width / og_x
    y_scale = height / og_y

    resized = cv2.resize(src=image, dsize=None, fx=x_scale, fy=y_scale)
    # for half the scale
    # resized = cv2.resize(src=image, dsize=None, fx=0.5, fy=0.5)
    if show:
        display_image("RESIZED IMAGE", resized)
    return resized


def draw_shapes_N_text(shape: str, image, input_text: Optional[str] = None):
    shapes = {
        "circle": cv2.circle,
        "rectangle": cv2.rectangle,
        "line": cv2.line,
        "text": cv2.putText,
    }
    if shape not in shapes.keys():
        return

    resized = resize_it(500, 500, image)
    center = resized.shape[0] // 2, resized.shape[1] // 2
    thickness = 2

    if shape == "circle":
        radius = 69
        color = (255, 0, 0)
        shapes[shape](resized, center, radius, color, thickness)

    elif shape == "rectangle":
        top_left = (200, 200)
        bottom_right = (500, 500)
        color = (0, 255, 0)
        shapes[shape](resized, top_left, bottom_right, color, thickness)

    elif shape == "line":
        start = (0, 0)
        end = (900, 50)
        color = (0, 0, 255)
        shapes[shape](resized, start, end, color, thickness)

    elif shape == "text" and input_text:
        origin = tuple([center[0], list(center)[1] // 2])
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1
        color = (0, 69, 45)
        shapes[shape](resized, input_text, origin, font, font_scale, color, thickness)

    display_image(f"{shape.upper()} DRAWN", resized)


def translate_it(image, tx: int, ty: int):

    resized = resize_it(500, 500, image)

    rows, cols = resized.shape[:2]
    translation_matrix = np.float32(np.array([[1, 0, tx], [0, 1, ty]]))
    translated = cv2.warpAffine(resized, np.asarray(translation_matrix), (cols, rows))

    display_image("Translation window", translated)


def rotate_it(image, angle: float, scale: float = 1.0):
    resized = resize_it(500, 500, image)
    # for now center of the image itself
    center = resized.shape[0] // 2, resized.shape[1] // 2

    rows, cols = resized.shape[:2]
    # gives the matrix for rotation
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # actually rotating it by appling transformation
    rotated = cv2.warpAffine(resized, matrix, (cols, rows))

    display_image("Rotation window", rotated)


def color_space_conversion(image, color_code: int):
    resized = resize_it(500, 500, image)
    converted = cv2.cvtColor(resized, color_code)

    display_image("OG image", resized)
    display_image("Converted image", converted)


def main():

    image = read_image("pink_chair.png")
    display_image("Image display", image)
    capture_from_webcam()

    # providing the desired size here
    resize_it(500, 500, image, show=True)

    # function examples
    draw_shapes_N_text("circle", image)
    draw_shapes_N_text("rectangle", image)
    draw_shapes_N_text("line", image)
    draw_shapes_N_text("text", image, "Pink chair it is")

    translate_it(image, -100, 200)
    rotate_it(image, 45.0)

    # changes to grayscale
    color_space_conversion(image, cv2.COLOR_RGB2GRAY)
    # changes to hsv
    color_space_conversion(image, cv2.COLOR_RGB2HSV)


if __name__ == "__main__":
    main()
