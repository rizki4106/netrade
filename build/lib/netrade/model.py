from torch import nn
import torch

class NetradeV2(nn.Module):
  """Netrade AI trading assistant
  Args:
    chart_channels : int -> image's channels in chart images
    candle_chanels : int -> image's channels in candle images
    hidden_units : int -> numbers of hidden unit in hidden layer
    output_features : int -> numbers of output ( number of classes )
  """

  def __init__(self,
               chart_channels : int = 3, 
               candle_channels : int = 3, 
               hidden_units : int = 10, 
               output_features : int = 2):

    super().__init__()

    # input layer for chart pattern
    self.chart_input_layer = nn.Sequential(
        nn.Conv2d(in_channels=chart_channels,
                  out_channels=hidden_units,
                  kernel_size=(2,2),
                  stride=1,
                  padding="valid")
    )

    # convolutional neural net block
    self.block_1_chart = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=(2,2),
                  stride=1,
                  padding="valid"),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=(2,2),
                  stride=1,
                  padding="valid"),
        
        nn.ReLU(),
        # nn.MaxPool2d(kernel_size=2)
    )

    self.block_2_chart = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=(2,2),
                  stride=1,
                  padding="valid",
                  ),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=(2,2),
                  stride=1,
                  padding="valid",
                  ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten()
    )

    # candle stick input layer
    self.candle_input_layer = nn.Sequential(
        nn.Conv2d(in_channels=candle_channels,
                  out_channels=hidden_units,
                  kernel_size=(2,2),
                  stride=1,
                  padding="valid"
        )
    )

    self.block_1_candle = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=(2,2),
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=(2,2),
                  stride=1,
                  padding=1),
        
        nn.ReLU(),
        # nn.MaxPool2d(kernel_size=2)
    )

    self.block_2_candle = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=(2,2),
                  stride=1,
                  padding=1,
                  ),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=(2,2),
                  stride=1,
                  padding=1,
                  ),
        nn.ReLU(),
        # nn.MaxPool2d(kernel_size=2),
        nn.Flatten()
    )

    # output layer
    self.classifier = nn.Sequential(
        nn.Linear(in_features=458360,out_features=output_features)
    )


  def forward(self, chart, candle):
    """Do the forward pass

    Args:
      chart : torch tensor -> array pixel of images from chart pattern
      candle : torch tensor -> array pixel of images from candle stick pattern
    """

    # forward pass chart pattern
    c = self.chart_input_layer(chart)
    c = self.block_1_chart(c)
    c = self.block_2_chart(c)

    # forward pass for candle stick pattern
    s = self.candle_input_layer(candle)
    s = self.block_1_candle(s)
    s = self.block_2_candle(s)

    # merge 2 returns
    merge = torch.cat([c, s], dim=1)

    # return output layer
    out = self.classifier(merge)
    return out