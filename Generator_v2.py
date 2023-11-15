import random
from PIL import Image, ImageDraw, ImageFont
from definitions import color_options

# Random generator for Shape, Color and Symbol


def imageGenerate(filename, img_h, img_w, color, shape, symbol, color2):
    """
    Generates image based on specified filename.
    """
    # check for error in command-line in capitalization
    selected_shape = shape.capitalize()
    selected_symbol = symbol.capitalize()
    color = color.capitalize()
    color2 = color2.capitalize()

    selected_color = color_options[color]
    selected_secondary = color_options[color2]

    print(selected_shape)
    print(color)
    print(selected_symbol)
    print(color2)

    color_values_list = list(color_options.values())
    if selected_color == selected_secondary:
        exit(1)

    gray = (128, 128, 128)

    # ('RGB', (100, 100), selected_color)
    img = Image.new('RGB', (img_h, img_w), gray)

    draw = ImageDraw.Draw(img)
    center = (0.5*img_w, 0.5*img_h)
    radius = 0.4*img_w

    if selected_shape == "Circle":
        draw.ellipse([center[0]-radius, center[1]-radius,
                     center[0]+radius, center[1]+radius], fill=selected_color)
    elif selected_shape == "Semi-circle":
        center = (0.5*img_w, 0.35*img_h)
        draw.pieslice([center[0]-radius, center[1]-radius, center[0] +
                      radius, center[1]+radius], 0, 180, fill=selected_color)
    elif selected_shape == "Quarter_circle":
        center = (0.325*img_w, 0.35*img_h)
        draw.pieslice([center[0]-radius, center[1]-radius, center[0] +
                      radius, center[1]+radius], 0, 90, fill=selected_color)
    elif selected_shape == "Star":
        # Define points for a star
        points = [
            (0.5*img_w, 0.1*img_h),
            (0.61*img_w, 0.5*img_h),
            (1.0*img_w, 0.5*img_h),
            (0.7*img_w, 0.7*img_h),
            (0.8*img_w, 1.0*img_h),
            (0.5*img_w, 0.8*img_h),
            (0.2*img_w, 1.0*img_h),
            (0.3*img_w, 0.7*img_h),
            (0, 0.5*img_h),
            (0.39*img_w, 0.5*img_h)
        ]
        draw.polygon(points, fill=selected_color)
    elif selected_shape == "Pentagon":
        # Define points for a regular pentagon
        points = [
            (.50 * img_w, img_h * .05),
            (img_w * 0.95, img_h * .45),
            (.80 * img_w, img_h * 0.95),
            (.20 * img_w, img_h * 0.95),
            (img_w * .05, img_h * .45),
        ]
        draw.polygon(points, fill=selected_color)
    elif selected_shape == "Triangle":
        # Define points for an equilateral triangle
        triangle = [
            (img_w * .50, img_h * .10),
            (img_w * .90, img_h * .90),
            (img_w * .10, img_h * .90),
        ]
        draw.polygon(triangle, fill=selected_color)
    elif selected_shape == "Rectangle":
        # Define points for a rectangle
        draw.rectangle([.2 * img_w, .2 * img_h, .80 *
                       img_w, .80 * img_h], fill=selected_color)
    elif selected_shape == "Cross":
        # Define points for a cross
        draw.rectangle([img_w*.20, img_h*.40, img_w*.80,
                       img_h*.60], fill=selected_color)
        draw.rectangle([img_w*.40, img_h*.20, img_w*.60,
                       img_h*.80], fill=selected_color)

    font = ImageFont.truetype("Roboto/Roboto-Black.ttf", size=50)
    draw.text((img_h * 0.465, img_w * 0.465),
              selected_symbol, selected_secondary, font=font)

    img.save(filename)
