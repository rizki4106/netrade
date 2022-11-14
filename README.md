# Netrade
Netrade is an AI trading assistant with human trader approach

## Table of contents
1. [How it works](#how-it-works)
2. [Performance Result](#performance)
3. [Installation](#installation)
4. [Usage](#usage)
    1. [Training](#training)
4. [Inference Mode or Testing](#inference-mode--testing)
    1. [Prepare Real-time Data](#real-time-data)
    2. [Run in real world and real-time data](#final-preparation)

## How it works <div id='how-it-works'/>
This AI model will predict the price will `go up` or `go down` based on chart pattern and candlestick pattern <br/>

The data is available [here](https://github.com/rizki4106/netrade/releases)

### Chart Pattern
Chart pattern is enough for analyzing price will go up or go down as usualy traders do, here's some example:

![chart-pattern](https://cdn.heelmet.com/files/chart-pattern.png)

### Candle Stick
Bechause of chart pattern has a limitation we need candle stick pattern to decide when we should buy or sell, here is some example

*first row is `bearish` candle stick pattern and the second row `bulish` candle stick*

![chart-pattern](https://cdn.heelmet.com/files/candle_chart.png)

### Performance
We've been tested this model in 1 week and here's the result:

* model accuracy & loss

![chart-pattern](https://cdn.heelmet.com/files/model_performance_v2.png)

* profit

![chart-pattern](https://cdn.heelmet.com/files/profit_model.png)

* Win & loss rate ( 61% win 39% loss )

![chart-pattern](https://cdn.heelmet.com/files/win_and_loss_rate.png)

# Installation

* github
    ```bash
    git clone https://github.com/rizki4106/netrade.git
    ```
    ```bash
    cd netrade && pip3 install torch torchmetrics scikit-image Pillow torchvision
    ```
* pypi
    ```bash
    pip3 install netrade
    ```

# Usage

This step devided into 2 step

## Training

download pre-trained model here [https://github.com/rizki4106/netrade/releases/](https://github.com/rizki4106/netrade/releases/) but
if you want to train with your own data, here's the step

* Prepare The data <br/>
You should put your image in this pattern:

    ```text
    chart:
    ----up:
    ------image1.png
    ------image2.png
    ----down:
    ------image1.png
    ------image2.png
    candle:
    ------image1.png
    ------image2.png
    ----down:
    ------image1.png
    ------image2.png
    ```
* Make csv file that contain this field

    | chart_name | candle_name | path | label |
    |-|-|-| -|
    | filename.png | filename.png | down | 0 |
    | filename.png | filename.png | up | 1 |
    | filename.png | filename.png | down | 0 |
    | filename.png | filename.png | down | 0 |
    | filename.png | filename.png | up | 1 |

    you can do it by using data preprocessing helper easly

    ```python
    from netrade.data import DataPreprocessing

    # initialize class
    chart = "/path/to/somwhere/chart/"
    candle = "/path/to/somewhere/candle"

    prep = DataPreprocessing(chart_path=chart, candle_path=candle)

    # create dataframe
    frame = prep.create_frame()
    frame.head()
    ```
* Create image transformer
    ```python
    from torchvision import transforms

    # this is for chart pattern
    chart_transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # this is for candlestick pattern
    candle_transformer = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    ```
* Load your data with data loader
    ```python
    from netrade.data import NetradeDataLoader

    # supposed you have created csv file like i mention above
    frame = pd.read_csv("file_training.csv")

    # load data and turn it into tensor
    train_data = NetradeDataLoader(
        chart_dir="/path/to/root/chart-pattern/",
        candle_dir="/path/to/root/candle-stick/",
        frame=frame,
        chart_transform=chart_transformer,
        candle_transform=candle_transformer
    ) # this data loader will return [chart image, candle image and labels]
    ```
* Create bathes
    ```python
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True
    )
    ```
* Run training loop

    ```python
    from netrade.core import Netrade
    import torch

    # initialize the model
    netrade = Netrade()

    # run training
    model, history = netrade.train(X_train=dataloader, epochs=10)

    # model is pure pytorch nn module that has been trained with your own data you can check it model.parameters()
    # history is the result from training loop

    print(history)

    # save the model's state
    torch.save(model.state_dict(), "name-it.pth")
    ```
## Inference mode / Testing
### Real-time Data
If you want to use this model in real time data, you should prepare the comodity price history i.e `bitcoin` or `tesla` stock price. in this example I'll be using `yfinance` to grab the historical data

```python
from netrade.data import data_creation
import yfinance as yf
import matplotlib.pyplot as plt

#
ticker = yf.Ticker('BTC-USD')
data = ticker.history(period="7d", interval="15m")

# create chart image from realtime history data
# take the last 50 candle stick bar
chart_image = data_creation.create_image(data=data[-50:, :])

# take the last 3 candle stick bar
candle_image = data_creation.create_image(data=data[-3:, :])

# create_image returns PIL image class that you can use directly with pytorch

# you can show it by the way
plt.imshow(chart_image)
plt.imshow(candle_image)
```

### Final Preparation
It's time to predict real - time price, let's put everything together

```python
from netrade.data import data_creation
from netrade.core import Netrade
from torchvision import transforms
from PIL import Image
import yfinance as yf

# initialize the model
netrade = Netrade(saved_state_path="path-to-saved-state.pth")

# create image transformer
# this is for chart pattern
chart_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# this is for candlestick pattern
candle_transformer = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# load realtime data
ticker = yf.Ticker('BTC-USD')
data = ticker.history(period="7d", interval="15m")

# create chart image from real-time data
chart_image = data_creation.create_image(data=data[-50:, :])
candle_image = data_creation.create_image(data=data[-3:, :])

# turn image into tensor
chart_image = chart_transformer(chart_image)
candle_image = candle_transformer(candle_image)

# run prediction
preds = netrade.predict(chart_image=chart_image, candle_image=candle_image)

# print the result
print(preds.argmax(1)) # 0 price will go down, 1 price will go up
```
