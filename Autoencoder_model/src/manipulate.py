import transforms
import yaml
import numpy as np
from PIL import ImageOps, ImageFilter, Image
import io

from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank


class Manipulator:
    """Manipulate PIL images with manipulations according to config file.
    Args:
        config_path (path-like object): path to config file. See example.yaml for usage.
    """

    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def __call__(self, img):
        """Apply manipulations to image.
        Args:
            img (PIL Image): Image to be manipulated.
        Returns:
            PIL Image: Anchor Image (128x128)
            PIL Image: Manipulated Image (128x128)
        """
        cfg = self.config
        anchor = transforms.center_crop(img, 128).copy()

        # flipping
        if np.random.rand() < cfg["p_horizontal_flip"]:
            img = transforms.hflip(img)
        if np.random.rand() < cfg["p_vertical_flip"]:
            img = transforms.vflip(img)

        # perspective
        if "perspective_range" in cfg:
            width, height = img.size
            startpoints = np.array([(0, 0), (0, height), (width, height), (width, 0)])
            rho = np.random.randint(*cfg["perspective_range"], startpoints.shape)
            endpoints = startpoints + rho
            img = transforms.perspective(img, startpoints, endpoints)

        # affine
        if "affine" in cfg:
            scale = np.random.uniform(*cfg["affine"]["scale_range"])
            rotation = np.random.uniform(*cfg["affine"]["rotation_range"])
            translate = list(np.random.uniform(*cfg["affine"]["translation_range"], 2))
            img = transforms.affine(img, rotation, translate, scale, shear=0)

        # gamma
        if "gamma_range" in cfg:
            img = transforms.adjust_gamma(img, np.random.uniform(*cfg["gamma_range"]))

        # hue
        if "hue" in cfg:
            img = transforms.adjust_hue(img, np.random.uniform(*cfg["hue_range"]))

        # brightness
        if "brightness_range" in cfg:
            img = transforms.adjust_brightness(
                img, np.random.uniform(*cfg["brightness_range"])
            )

        # contrast
        if "contrast_range" in cfg:
            img = transforms.adjust_contrast(
                img, np.random.uniform(*cfg["contrast_range"])
            )

        if "global_hist_p" in cfg and np.random.rand() < cfg["global_hist_p"]:
            img = (exposure.equalize_hist(np.array(img)) * 255).astype(np.uint8)
            img = Image.fromarray(img)

        elif "local_hist_p" in cfg and np.random.rand() < cfg["local_hist_p"]:
            selem = disk(10)
            img = (rank.equalize(np.array(img), selem=selem) * 255).astype(np.uint8)
            img = Image.fromarray(img)

        if np.random.rand() < cfg["p_invert"]:
            img = ImageOps.invert(img)

        if "p_blur" in cfg and np.random.rand() < cfg["p_blur"]:
            img = img.filter(ImageFilter.GaussianBlur(radius=2))

        if "jpeg_p" in cfg and np.random.rand() < cfg["jpeg_p"]:
            buffer = io.BytesIO()
            img.save(buffer, "JPEG", quality=50)
            img = Image.open(buffer)

        if "p_noise" in cfg and np.random.rand() < cfg["p_noise"]:
            img = np.array(img)
            img = img + np.random.normal(loc=0, scale=16, size=img.shape)
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)

        # grayscale
        if "grayscale" in cfg:
            img = transforms.to_grayscale(img, num_output_channels=cfg["grayscale"])
            anchor = transforms.to_grayscale(
                anchor, num_output_channels=cfg["grayscale"]
            )

        img = transforms.center_crop(img, 128)
        return anchor, img


if __name__ == "__main__":
    m = Manipulator("example.yaml")
    from skimage import data
    from PIL import Image
    import matplotlib.pyplot as plt

    img = Image.fromarray(data.astronaut())
    plt.imshow(img)
    plt.show()

    anchor, same = m(img)
    plt.imshow(same)
    plt.show()

    plt.imshow(anchor)
    plt.show()
