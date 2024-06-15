import cv2


def transparent_rectangle(image, opacity: int, *args):
    overlay = image.copy()

    cv2.rectangle(overlay, *args)

    image = cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)

    return image


def centroid(x, y, w, h):
    x1 = w / 2
    y1 = h / 2

    cx = x + x1
    cy = y + y1

    return int(cx), int(cy)
