from PIL import ImageFont, ImageDraw


def text_center(image, selected_font, size, text, Y, color=(39, 37, 37)):
    image_w, _image_h = image.size
    image_to_draw = ImageDraw.Draw(image)
    selected_font = ImageFont.truetype(selected_font, size=size)
    text_w, _text_height = image_to_draw.textsize(text, font=selected_font)
    image_to_draw.text(
        xy=((image_w - text_w) / 2, Y),
        text=text,
        fill=color,
        font=selected_font,
    )
