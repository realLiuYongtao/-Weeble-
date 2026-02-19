from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

def add_chinese_text(img, text, position, textColor=(0, 0, 0), textSize=5, digital=None):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("SGTCN/fonts/MSYH.ttc", textSize, encoding="utf-8")
    # 绘制文本
    if digital is not None:
        text = text+str(digital)+'%'
    draw.text(position, text, textColor, font=fontStyle)
    draw.text((0,0),"按Q或Esc键退出",(0,0,0),font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

