from torch.utils.data import Dataset
from PIL import Image
import mplfinance as mf
import numpy as np
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
    chart_image = Image.fromarray(Image.open(chart_dirs))

    # convert chart image from others to RGB
    chart_image = chart_image.convert("RGB")

    # join path between root dir and candle image name
    candle_dirs = os.path.join(self.candle_dir, self.data_frame.iloc[idx, 2], self.data_frame.iloc[idx, 1])

    # read candle pattern image
    candle_image = Image.fromarray(Image.open(candle_dirs))
    
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

# Data preprocessing section
# The code below do the preprocesing needs such as create data frame for training needs, shuffling and etc

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

# data creation section
# This section below shows the code base for data creation needs
# such as create chart pattern and candle stick pattern image

class DataCreation:
  """
  This class used for create data needs
  """
  def create_image(self, data : list = []):
    """Create candle stick or chart pattern image and return it as PIL Image class.

    Args:
        data : pandas data frame -> list historycal data in pandas data frame format from yfinance or other

    Returns:
        image : PIL Image -> PIL image class
    """

    # make candlestick style just as the same with trading view color
    mc = mf.make_marketcolors(up='#26A69A', edge='inherit', down='#EF5350', wick={"up" : '#26A69A', 'down': '#EF5350'})

    # configuring figure style
    s  = mf.make_mpf_style(gridstyle="", marketcolors=mc, edgecolor="#ffffff")

    # create figure
    fig = mf.figure(style=s)
    ax = fig.add_subplot(1, 1, 1)

    # remove x and y label
    ax.axis(False)

    # create candle stick
    mf.plot(data,ax=ax,type="candle")

    # draw candle stick into canvas
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    candle_arr = np.array(fig.canvas.renderer.buffer_rgba())

    # convert numpy array to PIL image
    img = Image.fromarray(candle_arr).convert("RGB")

    return img

if __name__ != "__main__":

  # initiate data creation helper
  data_creation = DataCreation()