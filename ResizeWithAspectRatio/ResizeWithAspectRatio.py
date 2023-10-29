import cv2

# Formulas Used:
# ratio = new_width / image_width
# or
# ratio = new_height / image_height
# scaled_image_width = image_width * r
# scaled_image_height = image_height * r


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)
