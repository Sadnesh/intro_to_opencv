import cv2
from typing import Optional


def read_image(image_path: str):
    return cv2.imread(image_path)


# display image
def display(image_path: str):
    print("Displaying image now")
    image = read_image(image_path)

    cv2.imshow("Chair", image)
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


def resize_it(width: float, height: float, image_path: str, show: bool = False):
    image = read_image(image_path)

    # gives the image size
    og_y, og_x = image.shape[:2]

    x_scale = width / og_x
    y_scale = height / og_y

    resized = cv2.resize(src=image, dsize=None, fx=x_scale, fy=y_scale)
    # for half the scale
    # resized = cv2.resize(src=image, dsize=None, fx=0.5, fy=0.5)
    if show:
        cv2.imshow("RESIZED IMAGE", resized)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return resized


def draw_shapes_N_text(shape: str, image_path: str, input_text: Optional[str] = None):
    shapes = {
        "circle": cv2.circle,
        "rectangle": cv2.rectangle,
        "line": cv2.line,
        "text": cv2.putText,
    }
    if shape not in shapes.keys():
        return

    # image = read_and_display(image_path)
    image = resize_it(500, 500, image_path)
    center = image.shape[0] // 2, image.shape[1] // 2
    thickness = 2

    if shape == "circle":
        radius = 69
        color = (255, 0, 0)
        shapes[shape](image, center, radius, color, thickness)

    elif shape == "rectangle":
        top_left = (200, 200)
        bottom_right = (500, 500)
        color = (0, 255, 0)
        shapes[shape](image, top_left, bottom_right, color, thickness)

    elif shape == "line":
        start = (0, 0)
        end = (900, 50)
        color = (0, 0, 255)
        cv2.line(image, start, end, color, thickness)

    elif shape == "text" and input_text:
        origin = tuple([center[0], list(center)[1] // 2])
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1
        color = (0, 69, 45)
        shapes[shape](image, input_text, origin, font, font_scale, color, thickness)

    cv2.imshow(f"{shape.upper()} DRAWN", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    display("pink_chair.png")
    capture_from_webcam()

    # providing the desired size here
    resize_it(500, 500, "pink_chair.png", show=True)

    # function examples
    draw_shapes_N_text("circle", "pink_chair.png")
    draw_shapes_N_text("rectangle", "pink_chair.png")
    draw_shapes_N_text("line", "pink_chair.png")
    draw_shapes_N_text("text", "pink_chair.png", "Pink chair it is")


if __name__ == "__main__":
    main()
