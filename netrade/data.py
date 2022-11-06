from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import torch
import os
import pandas as pd

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

class DataPreprocessing:
  """
  This class used for data pre-processing and manipulating needs
  """

  def __init__(self, chart_path, candle_path):

    self.chart_path = chart_path
    self.candle_path = candle_path

  def create_frame(self, shuffle : bool = False):
    """Generate pandas data frame for use in netrade data loader

    Args: 
      chart_path : str -> path to chart image data
      candle_path : str -> path to candle stick image data

    Returns:
      frame : pandas dataframe -> return pandas data frame
    """

    result = []

    # loop to folder for each classes e.g candlestick -> down
    for classes in os.listdir(self.candle_path):

      # loop all files inside each classes
      for candle in os.listdir(os.path.join(self.candle_path, classes)):

        # loop all chart data
        for chart in os.listdir(os.path.join(self.chart_path, classes)):
          
          context = {}
          context['chart_name'] = chart
          context['candle_name'] = candle
          context['path'] = classes
          context['label'] = 0 if classes == "down" else 1

          # concat all data to result variable
          result.append(context)
    
    frame = pd.DataFrame(result)

    # shuffle data if needed
    if shuffle:
      frame = frame.sample(frac=1)

    return frame