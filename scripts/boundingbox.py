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