# Netrade
Netrade is an AI trading assistant with human trader approach

## How it works
This AI model will predict the price will `go up` or `go down` based on chart pattern and candlestick pattern

### Chart Pattern
Chart pattern is enough to price will go up or go down as usualy traders do, here's some example chart bulish pattern :

chart bearish pattern :

### Candle Stick
Bechause of chart pattern has a limitation we need candle stick pattern to decide when we should buy or sell, here is some example

bulish candlestick pattern :

bearish candlestick pattern:
### Performance
We've been tested this model in 1 week and here's the result:

* model accuracy & loss

* profit

* Win & loss rate

# Installation

* github
    ```bash
    git clone https://github.com/rizki4106/netrade.git
    ```
    ```bash
    cd netrade && pip3 install torch torchmetrics scikit-image Pillow
    ```
* pypi
    ```bash
    pip3 install netrade
    ```

# Usage

This step devided into 2 step

## Training

If you want to train with your own data, here's the step

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