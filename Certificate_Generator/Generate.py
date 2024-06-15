from _Libraries.Certificate_Generator import text_center
from PIL import Image

text1 = "MAPÚA UNIVERSITY"
text2 = "Certificate of Completion"
text3 = "PAMBID, LUIS DANIEL A."
text4 = "March 19, 2019"
text5 = "TIME IN: 11:59 PM"
text6 = "TIME OUT: 12:00 PM"
text7 = "DURATION: 03:00:00 MINUTES"

image = Image.open("Cert_Template.png")
text_center(image, "Fonts/Nunito-Light.ttf", 80, text1, 20)
text_center(image, "Fonts/vivaldi.ttf", 60, text2, 150)
text_center(image, "Fonts/cambria.ttf", 60, text3, 250)
text_center(image, "Fonts/vivaldi.ttf", 60, text4, 350)
text_center(image, "Fonts/cambria.ttf", 40, text5, 450)
text_center(image, "Fonts/cambria.ttf", 40, text6, 500)
text_center(image, "Fonts/cambria.ttf", 40, text7, 550)
image.save("Output.png", "PNG", resolution=100.0)
