import os
import cv2 as cv


def load_image(image_path: str):
    return cv.imread(image_path)


def display(window_name: str, image):
    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def convert_to_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def resize_image(image):
    height = int(input("Enter the height of the final image: "))
    width = int(input("Enter the width of the final image: "))

    return cv.resize(image, (height, width))


def rotate_image(image):
    h, w = image.shape[:2]
    center = w // 2, h // 2
    angle = int(input("Enter the angle of rotation: "))

    rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)

    return cv.warpAffine(image, rot_mat, (w, h))


def detect_edges(image):

    threshold1 = int(
        input("Enter the first threshold below which the values are discarded: ")
    )
    threshold2 = int(
        input(
            "Enter the second threshold above which the values are considered strong: "
        )
    )
    return cv.Canny(image, threshold1, threshold2)


def add_text(image):

    text = input("Enter the text you want to add in the image: ")
    print("Where do you want your text?")
    x, y = int(input("Enter the x position: ")), int(input("Enter the y position: "))

    return cv.putText(image, text, (x, y), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 2)


def draw_rectangle(image):

    print(
        "Okay, hear me out! You need to provide (x1,y1) and (x2,y2) to draw a rectangle, SO"
    )
    x1, y1 = int(input("Enter the x1 position: ")), int(
        input("Enter the y1 position: ")
    )
    x2, y2 = int(input("Enter the x2 position: ")), int(
        input("Enter the y2 position: ")
    )

    return cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


def save_image(image):
    file_name = input("Enter the filename where you want to save the image: ")
    if file_name.strip().lower() == "save here":
        cv.imwrite(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "edited_image.png"),
            image,
        )
        return
    cv.imwrite(file_name, image)


def main():
    image = None
    while image is None:
        print("Welcome to my Photo Editor")
        image_path = input(
            "Enter you image path (or simply name, if in same dir) to start working on it: "
        )
        image = load_image(image_path)
        print("Enter a valid image path or image name") if image is None else ""

    while True:

        print(
            "Options \n\
              1. Convert to Grayscale \n\
              2. Resize image   \n\
              3. Rotate image   \n\
              4. Detect Edge    \n\
              5. Add Text   \n\
              6. Draw rectangle \n\
              7. Save image \n\
              8. Exit"
        )

        choice = input("Enter the option number you want to perform: ")

        match choice:
            case "1":
                display("Grayscale Image", convert_to_grayscale(image))

            case "2":
                display("Resized Image", resize_image(image))

            case "3":
                display("Rotated Image", rotate_image(image))

            case "4":
                display("Edge detected Image", detect_edges(image))

            case "5":
                display("Text added Image", add_text(image))

            case "6":
                display("Rectangle drawn Image", draw_rectangle(image))

            case "7":
                save_image(image)
                print("Image saved sucessfully!")

            case "8":
                print("Thanks for using the Photo Editor!!")
                break

            case _:
                print("Please enter a valid option number.")


if __name__ == "__main__":
    main()
