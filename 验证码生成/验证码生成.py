import PIL
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

for i in range(10):
    # 随机字母:
    def rndChar():
        return chr(random.randint(48, 57))  # 65-90

    # 随机颜色1:
    def rndColor():
        return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))


    # 随机颜色2:
    def rndColor2():
        return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))


    # 240 x 60:
    width = 100 * 4
    height = 100
    image = Image.new('RGB', (width, height), (255, 255, 255))

    # 创建Font对象:
    font = ImageFont.truetype('C:\Windows\Fonts\Arial.ttf', 36)
    # font = ImageFont.truetype("symbol.ttf", 16, encoding="symb")

    # 创建Draw对象:
    draw = ImageDraw.Draw(image)
    # 填充每个像素:
    for x in range(width):
        for y in range(height):
            draw.point((x, y), fill=rndColor())

    # 输出文字:
    for t in range(4):
        draw.text((100 * t + 30, 30), rndChar(), font=font, fill=rndColor2())

    # 模糊:
    image = image.filter(ImageFilter.BLUR)
    image.save('E://pytorch/验证码生成/data/验证码图片{}.jpg'.format(i), 'jpeg')

