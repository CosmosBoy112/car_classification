"""Image loading, optional bbox crop, train augmentation, and degradation.

Pipeline for one image:
    load RGB -> (optional) bbox crop with margin -> train aug or plain resize to 224x224
             -> (optional) degradation -> tensor + ImageNet normalization

Degradation is applied after resizing to 224 so "64 px" always means the same
loss of detail, whatever the original photo size was.
"""
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

import config as C


def read_split(name: str) -> pd.DataFrame:
    """name in {train, val, test}. Splits are only ever read, never rebuilt."""
    return pd.read_csv(C.SPLITS_DIR / f"{name}.csv")


def crop_box(img: Image.Image, row, margin: float = C.BBOX_MARGIN) -> Image.Image:
    bw, bh = row.x2 - row.x1, row.y2 - row.y1
    x1 = max(0, row.x1 - margin * bw)
    y1 = max(0, row.y1 - margin * bh)
    x2 = min(img.width, row.x2 + margin * bw)
    y2 = min(img.height, row.y2 + margin * bh)
    return img.crop((int(x1), int(y1), int(round(x2)), int(round(y2))))


class AspectAwareRandomResizedCrop:
    """RandomResizedCrop whose aspect range is relative to the input's own aspect.

    The torchvision version samples crops with aspect 3/4..4/3 in absolute terms,
    so on a 2:1 car photo it keeps a near-square window and throws away the front
    or back of the car. Here the range is (3/4..4/3) x the image's aspect.
    """

    def __init__(self, size, scale=C.RRC_SCALE, ratio=C.RRC_RATIO):
        self.size, self.scale, self.ratio = size, scale, ratio

    def __call__(self, img):
        aspect = img.width / img.height
        ratio = (self.ratio[0] * aspect, self.ratio[1] * aspect)
        i, j, h, w = T.RandomResizedCrop.get_params(img, self.scale, ratio)
        return TF.resized_crop(img, i, j, h, w, [self.size, self.size], antialias=True)


# ---------------------------------------------------------------- degradation
def downscale(img: Image.Image, res: int) -> Image.Image:
    """Simulate a low-resolution camera: shrink to res x res, then back up."""
    if res >= img.width:
        return img
    small = img.resize((res, res), Image.BILINEAR, reducing_gap=None)
    return small.resize(img.size, Image.BILINEAR)


def blur(img: Image.Image, sigma: float) -> Image.Image:
    return img if sigma <= 0 else img.filter(ImageFilter.GaussianBlur(radius=sigma))


def darken(img: Image.Image, factor: float) -> Image.Image:
    """factor 1.0 = unchanged, 0.25 = a quarter of the brightness."""
    return img if factor >= 1 else ImageEnhance.Brightness(img).enhance(factor)


def apply_degradation(img: Image.Image, degrade: dict | None) -> Image.Image:
    """degrade like {"res": 64} or {"blur": 2.0} or {"dark": 0.3}; keys can be combined."""
    if not degrade:
        return img
    if "res" in degrade:
        img = downscale(img, degrade["res"])
    if "blur" in degrade:
        img = blur(img, degrade["blur"])
    if "dark" in degrade:
        img = darken(img, degrade["dark"])
    return img


class RandomDegradation:
    """Train-time augmentation for robustness (analysis 7c).

    Each corruption is applied independently with probability p, so most
    images still show clean detail and the model does not forget it.
    """

    def __init__(self, p=0.3, res_range=(24, 160), blur_range=(0.5, 3.0), dark_range=(0.2, 0.8)):
        self.p, self.res_range, self.blur_range, self.dark_range = p, res_range, blur_range, dark_range

    def __call__(self, img):
        if random.random() < self.p:
            img = downscale(img, random.randint(*self.res_range))
        if random.random() < self.p:
            img = blur(img, random.uniform(*self.blur_range))
        if random.random() < self.p:
            img = darken(img, random.uniform(*self.dark_range))
        return img


# ---------------------------------------------------------------- dataset
class CarDataset(Dataset):
    def __init__(self, split: str | pd.DataFrame, crop: bool, train: bool = False,
                 degrade: dict | None = None, degrade_aug: bool = False, img_size: int = C.IMG_SIZE):
        self.df = read_split(split) if isinstance(split, str) else split.reset_index(drop=True)
        self.crop, self.train, self.degrade, self.img_size = crop, train, degrade, img_size

        if train:
            steps = [AspectAwareRandomResizedCrop(img_size), T.RandomHorizontalFlip(),
                     T.ColorJitter(**C.COLOR_JITTER)]
            if degrade_aug:
                steps.append(RandomDegradation())
        else:
            # Squash to a square instead of center cropping, so wide cars keep both ends
            steps = [T.Resize((img_size, img_size), antialias=True)]
        self.geometric = T.Compose(steps)
        self.to_tensor = T.Compose([T.ToTensor(), T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)])

    def __len__(self):
        return len(self.df)

    def load(self, idx: int) -> Image.Image:
        """Returns the PIL image right before tensor conversion (useful for galleries)."""
        row = self.df.iloc[idx]
        img = Image.open(C.IMAGE_ROOT / row.path).convert("RGB")
        if self.crop:
            img = crop_box(img, row)
        img = self.geometric(img)
        return apply_degradation(img, self.degrade)

    def __getitem__(self, idx):
        return self.to_tensor(self.load(idx)), int(self.df.label.iat[idx])


def seed_everything(seed: int = C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id):
    # Each worker gets a different but reproducible seed for Python's random module
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int = C.NUM_WORKERS):
    g = torch.Generator()
    g.manual_seed(C.SEED)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init, generator=g, drop_last=False,
    )
