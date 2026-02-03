from PIL import Image
import numpy as np
from pathlib import Path

def colour_swap(in_img, out_img, colours, range_tolerance):
    """
    Swap colors in an image based on a color mapping dictionary with tolerance.
    
    Parameters
    ----------
    in_img : str or Path
        Input image path
    out_img : str or Path
        Output image path (format determined by extension)
    colours : dict
        Mapping of source hex colors to target hex colors
        e.g., {'#000000': '#140E00', '#FFFFFF': '#E8D9A8'}
    range_tolerance : int
        Tolerance for color matching (±range for each RGB channel)
        e.g., 10 means colors within ±10 of target will match
    """
    # Load image
    img = Image.open(in_img).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    
    # Convert hex colors to RGB tuples
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    colour_map = {hex_to_rgb(src): hex_to_rgb(tgt) 
                  for src, tgt in colours.items()}
    
    # Create output array
    output_array = img_array.copy()
    
    # For each source color, find and replace matching pixels
    for src_rgb, tgt_rgb in colour_map.items():
        # Calculate color distance for all pixels
        distance = np.abs(img_array - np.array(src_rgb))
        
        # Mask where all RGB channels are within tolerance
        mask = np.all(distance <= range_tolerance, axis=2)
        
        # Apply replacement
        output_array[mask] = tgt_rgb
    
    # Convert back to image and save
    output_img = Image.fromarray(output_array.astype(np.uint8), mode='RGB')
    output_img.save(out_img)
    print(f"Color swap complete: {out_img}")



def colour_swap_closest(in_img, out_img, colours, range_tolerance):
    """
    Version that assigns each pixel to the closest matching source color
    within tolerance, avoiding conflicts.
    """
    img = Image.open(in_img).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    colour_map = {hex_to_rgb(src): hex_to_rgb(tgt) 
                  for src, tgt in colours.items()}
    
    output_array = img_array.copy()
    
    # Calculate distances to all source colors
    src_colors = np.array(list(colour_map.keys()))
    tgt_colors = np.array(list(colour_map.values()))
    
    # Reshape for broadcasting
    pixels = img_array.reshape(-1, 3)
    
    # Calculate Euclidean distance to each source color
    distances = np.sqrt(np.sum((pixels[:, np.newaxis, :] - src_colors[np.newaxis, :, :]) ** 2, axis=2))
    
    # Find closest color and check if within tolerance
    closest_idx = np.argmin(distances, axis=1)
    closest_dist = np.min(distances, axis=1)
    
    # Apply color swap where distance is within tolerance
    within_tolerance = closest_dist <= (range_tolerance * np.sqrt(3))  # Euclidean tolerance
    pixels[within_tolerance] = tgt_colors[closest_idx[within_tolerance]]
    
    output_array = pixels.reshape(img_array.shape)
    output_img = Image.fromarray(output_array.astype(np.uint8), mode='RGB')
    output_img.save(out_img)
    print(f"Color swap complete: {out_img}")
