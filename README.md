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

### Training

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
    ```