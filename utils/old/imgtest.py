import cv2

ROWS = 128
COLS = 128

def read_and_resize_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def write_image(img,file_path):
    cv2.imwrite(file_path,img)


def main():
    img = read_and_resize_image("/utils/ISIC_0015719.jpg")
    write_image(img, "/utils/resized/ISIC_0015719.jpg")


if __name__ == "__main__":
    main()