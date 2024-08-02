import cv2


# Read image and display it
def read_and_display(image_path: str):
    print("Displaying image now")
    image = cv2.imread(image_path)

    cv2.imshow("Chair", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image


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


def resize_it(width: float, height: float, image_path: str):
    image = read_and_display(image_path)

    # gives the image size
    og_y, og_x = image.shape[:2]

    x_scale = width / og_x
    y_scale = height / og_y

    resized = cv2.resize(src=image, dsize=None, fx=x_scale, fy=y_scale)
    # for half the scale
    # resized = cv2.resize(src=image, dsize=None, fx=0.5, fy=0.5)

    cv2.imshow("RESIZED IMAGE", resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    read_and_display("pink_chair.png")
    capture_from_webcam()

    # providing the desired size here
    resize_it(500, 500, "pink_chair.png")


if __name__ == "__main__":
    main()
