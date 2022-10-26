from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import torch
import os

class NetradeDataLoader(Dataset):

  def __init__(self, chart_dir, candle_dir, frame, chart_transform=None, candle_transform = None):
    """Data loader for training needs

    Args:
      chart_dir : str -> root directory to chart pattern data
      candle_dir : str -> root directory to candle stick pattern data
      frame : pandas data frame -> list of data in table format
      chart_transform : torchvision transforms -> image transformation
      candle_transform : torcvision transforms -> image transformation

    Returns:
      chart_tensor, candle_tensor, labels
    """

    super().__init__()

    self.data_frame = frame
    self.chart_dir = chart_dir
    self.candle_dir = candle_dir
    self.chart_transform = chart_transform
    self.candle_transform = candle_transform

  def __len__(self):

    return len(self.data_frame)

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()

    # join between root dir and chart image name
    chart_dirs = os.path.join(self.chart_dir, self.data_frame.iloc[idx, 2], self.data_frame.iloc[idx, 0])

    # read chart pattern image
    chart_image = Image.fromarray(io.imread(chart_dirs))

    # convert chart image from others to RGB
    chart_image = chart_image.convert("RGB")

    # join path between root dir and candle image name
    candle_dirs = os.path.join(self.candle_dir, self.data_frame.iloc[idx, 2], self.data_frame.iloc[idx, 1])

    # read candle pattern image
    candle_image = Image.fromarray(io.imread(candle_dirs))
    
    # convert image channel from others to rgb
    candle_image = candle_image.convert("RGB")

    # read labels and turn it into tensor
    labels = torch.tensor(self.data_frame.iloc[idx, -1].tolist())

    # check if there's a transforms
    if self.chart_transform:
      chart_image = self.chart_transform(chart_image)
    
    if self.candle_transform:
      candle_image = self.candle_transform(candle_image)

    return chart_image, candle_image, labels