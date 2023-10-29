import easyocr


def read_text_from_image(img):
    reader = easyocr.Reader(["en"])
    result = reader.readtext(img, detail=0)
    if result is not None and len(result) > 0:
        text = result[0]
        return text
    return ""


if __name__ == "__main__":
    text = read_text_from_image("Pictures/FOR_OCR.jpg")
    print(text)
