import dataclasses
import json
import os
import pprint
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches

from xml.etree import ElementTree

@dataclasses.dataclass
class Ink:
    """Represents a single ink, as read from an InkML file."""
    strokes: list[np.ndarray]
    annotations: dict[str, str]

def read_inkml_file(filename: str) -> Ink:
    """Simple reader for MathWriting's InkML files."""
    with open(filename, "r") as f:
        root = ElementTree.fromstring(f.read())

    strokes = []
    annotations = {}

    for element in root:
        tag_name = element.tag.removeprefix('{http://www.w3.org/2003/InkML}')
        if tag_name == 'annotation':
            annotations[element.attrib.get('type')] = element.text
        elif tag_name == 'trace':
            points = element.text.split(',')
            stroke_x, stroke_y, stroke_t = [], [], []
            for point in points:
                x, y, t = point.split(' ')
                stroke_x.append(float(x))
                stroke_y.append(float(y))
                stroke_t.append(float(t))
            strokes.append(np.array((stroke_x, stroke_y, stroke_t)))

    return Ink(strokes=strokes, annotations=annotations)

def save_ink_as_png(ink: Ink, *, figsize=(15, 10), linewidth=2, color='black', save_path=""):
    """Save ink as PNG with proper cleanup."""
    try:
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Turn off the axis and set white background
        plt.axis('off')
        fig.patch.set_facecolor('white')
        
        # Plot strokes
        for stroke in ink.strokes:
            plt.plot(stroke[0], stroke[1], linewidth=linewidth, color=color)
        
        # Configure plot settings
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tick_params(axis='both', which='both', length=0)
        
        # Remove spines
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        
        # Save the figure
        plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
        
    except Exception as e:
        print(f"Error processing {save_path}: {str(e)}")
    
    finally:
        # Clean up
        plt.close(fig)

def process_folder(folder_path, output_dir):
    """Process all files in the given folder with proper error handling."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    errors = []
    
    for root, dirs, files in os.walk(folder_path):
        total_files = len(files)
        
        for file in files:
            if file.endswith('.inkml'):  # Only process inkml files
                try:
                    file_path = os.path.join(root, file)
                    file_name = os.path.splitext(file)[0]
                    output_path = os.path.join(output_dir, f"{file_name}.png")
                    
                    # Skip if output file already exists
                    if os.path.exists(output_path):
                        continue
                    
                    ink = read_inkml_file(file_path)
                    save_ink_as_png(ink=ink, color='black', save_path=output_path)
                    
                    count += 1
                    if count % 100 == 0:  # Progress update every 100 files
                        print(f"Processed {count}/{total_files} files")
                        
                except Exception as e:
                    errors.append((file, str(e)))
                    print(f"Error processing {file}: {str(e)}")
    
    # Print summary
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {count} files")
    print(f"Errors encountered: {len(errors)}")
    
    if errors:
        print("\nFiles with errors:")
        for file, error in errors:
            print(f"- {file}: {error}")

# Example usage
CURRENT_DIR = os.getcwd()
MATHWRITING_ROOT_DIR = os.path.join(CURRENT_DIR, './mathwriting-2024')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'math-image-dataset/train-images')

train_folder = os.path.join(MATHWRITING_ROOT_DIR, 'train')
process_folder(train_folder, OUTPUT_DIR)