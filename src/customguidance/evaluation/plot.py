import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_image(image, title="Image", cmap=None, figsize=(8, 6)):
    """
    Affiche une PIL.Image avec Matplotlib.

    Args:
        image   : PIL.Image ou chemin vers un fichier (str)
        title   : titre affiché au-dessus de l'image
        cmap    : colormap (ex. 'gray') — None = auto-détection
        figsize : taille de la figure en pouces (largeur, hauteur)
    """
    # Chargement depuis un chemin fichier
    if isinstance(image, str):
        image = Image.open(image)

    # Conversion PIL → NumPy
    arr = np.array(image)

    # Auto-détection de la colormap
    if cmap is None and arr.ndim == 2:
        cmap = "gray"

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(arr, cmap=cmap)
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    plt.tight_layout()
    plt.show()