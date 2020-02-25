import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

font_size_min = 2
font_size_max = 6

word_length_min = 3
word_length_max = 9

vert_spacing_min = .05
vert_spacing_max = .1
horiz_spacing_min = .05
horiz_spacing_max = .3

alpha_min = 0.05
alpha_max = 0.2

theta_min = -60
theta_max = 60

def black_image(width, height):
    return Image.new('RGBA', (width, height), color='black')


def random_letter():
    offset = np.random.randint(0, 26)
    case = np.random.choice([65, 97])
    return chr(offset + case)


def random_word():
    word = ""
    length = np.random.randint(word_length_min, word_length_max)
    for i in range(length):
        word += random_letter()
    
    return word

def make_word_grid(side_length):

    font_size = np.random.randint(font_size_min, font_size_max)

    word = random_word()
    font = ImageFont.truetype("arial.ttf", font_size)
    word_width, word_height = font.getsize(word)[:2]

    vert_spacing = np.random.uniform(vert_spacing_min, vert_spacing_max) * side_length
    horiz_spacing = np.random.uniform(vert_spacing_min, vert_spacing_max) * side_length

    num_rows = int((side_length - vert_spacing) / (word_height + vert_spacing))
    num_cols = int((side_length - horiz_spacing) / (word_width + horiz_spacing))

    img = Image.new('RGB', (side_length, side_length), color='black')
    draw = ImageDraw.Draw(img)

    for i in range(num_rows):
        for j in range(num_cols):
            pos = (j * (word_width + horiz_spacing) + horiz_spacing, 
                   i * (word_height + vert_spacing) + vert_spacing) 
            draw.text(pos, word, fill=(255,255,255))
    
    return img

def make_watermark(width, height):
    # we are going to rotate the backdrop, 
    # so it must be as wide as the diagonal of the photo
    backdrop_length = round((width ** 2 + height ** 2) ** (1 / 2))

    backdrop = make_word_grid(backdrop_length)

    backdrop = backdrop.rotate(np.random.randint(theta_min, theta_max))
    backdrop.save('test2.png')

    backdrop = np.array(backdrop, dtype="float64")
    backdrop *= np.random.uniform(alpha_min, alpha_max)
    backdrop = backdrop.astype("uint8")

    width_start = int((backdrop_length - width) / 2)
    width_end = backdrop_length - width_start
    height_start = int((backdrop_length - height) / 2)
    height_end = backdrop_length - height_start

    backdrop = backdrop[height_start:height_end, width_start:width_end]

    return backdrop


def apply_watermark(img, watermark):
    result = img.copy().astype('int64')
    result += watermark
    result = result.flatten()
    result[result > 255] = 255
    return result.reshape(watermark.shape).astype('uint8')

def add_watermark(img):
    watermark = make_watermark(img.shape[1], img.shape[0])
    return apply_watermark(img, watermark)
    