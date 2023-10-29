from PIL import Image
from ResourceEmbed import resource_path

img = Image.open(resource_path("lion1.png"))
img.show()
