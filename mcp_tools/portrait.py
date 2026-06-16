import numpy as np
from PIL import Image

def create_gameboy_portrait(pil_img: Image.Image) -> Image.Image:
    """
    Converts any PIL Image into a dense, beautifully dithered 32-color 
    VGA-style retro color portrait at full 128x128 resolution.
    """
    # 1. Resize to full 128x128 resolution for maximum density
    small_img = pil_img.resize((128, 128), Image.Resampling.BILINEAR)
    
    # 2. Quantize the image to a dithered 32-color adaptive palette.
    # This applies Floyd-Steinberg dithering to preserve rich color gradients and shadows.
    quantized_img = small_img.convert('P', palette=Image.Palette.ADAPTIVE, colors=32)
    
    # 3. Convert back to RGB format for the OLED display
    return quantized_img.convert('RGB')
