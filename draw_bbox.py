
from PIL import Image , ImageDraw
im = Image.open('test.jpeg') 
drawSurface = ImageDraw.Draw(im)


drawSurface.line(((300,100),(300,600), (700,600), (700,100), (300,100)),fill = (255,0,0), width = 5)

im.show()
del drawSurface
#im.save('test.jpeg',format='jpeg')
