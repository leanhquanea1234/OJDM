#!/usr/bin/env python3
"""
Convert 128×64 monochrome PNG images to packed 1-bit byte data.

Reads all PNG files from a folder, converts each to row-major, MSB-first
packed bits, and saves the output as text files with the same name.
"""

import sys
from pathlib import Path
from PIL import Image


def png_to_packed_bytes(png_path: Path) -> str:
    """
    Convert a 128×64 monochrome PNG to packed 1-bit byte string.

    Parameters
    ----------
    png_path : Path
        Path to the PNG file.

    Returns
    -------
    str
        Bit string (e.g., '10101110...') representing packed pixels.
        Row-major, MSB-first within each byte.

    Raises
    ------
    ValueError
        If PNG is not exactly 128×64 pixels.
    """
    img = Image.open(png_path).convert("L")  # Convert to grayscale
    
    if img.size != (128, 64):
        raise ValueError(f"Expected 128×64, got {img.size}")
    
    pixels = img.getdata()
    bits = []
    
    # Process row-major: 64 rows of 128 pixels each
    for row in range(64):
        for col in range(128):
            pixel = pixels[row * 128 + col]
            # Threshold: pixel >= 128 → white (1), else black (0)
            bits.append("1" if pixel >= 128 else "0")
    
    return "".join(bits)


def convert_folder(folder_path: Path, output_folder: Path = None) -> None:
    """
    Convert all PNG files in a folder to packed byte strings.

    Parameters
    ----------
    folder_path : Path
        Folder containing PNG files.
    output_folder : Path, optional
        Where to save output files. Defaults to the same folder.
    """
    folder_path = Path(folder_path)
    output_folder = output_folder or folder_path
    output_folder.mkdir(parents=True, exist_ok=True)
    
    png_files = list(folder_path.glob("*.png"))
    if not png_files:
        print(f"No PNG files found in {folder_path}")
        return
    
    for png_file in png_files:
        try:
            bit_string = png_to_packed_bytes(png_file)
            output_file = output_folder / png_file.with_suffix("").name
            output_file.write_text(bit_string)
            print(f"{png_file.name} → {output_file.name}")
        except Exception as e:
            print(f"{png_file.name}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python png_to_bytes.py <input_folder> [output_folder]")
        sys.exit(1)
    
    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    convert_folder(input_folder, output_folder)

