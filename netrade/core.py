from netrade.model import NetradeV2
from torchmetrics import Accuracy
from torch import nn
import torch

# device diagnostic
device = "cuda" if torch.cuda.is_available() else "cpu"

class Netrade:

    def __init__(self, saved_state_path : str = None):
        """Predict cryptocurrency or stock price will go up or down
        Args:
            saved_state_path : str -> saved training state (.pth or .pt file)
        """

        # initialize model
        model = NetradeV2(
            chart_channels=3,
            candle_channels=3,
            hidden_units=10,
            output_features=2
        ).to(device)

        # load saved state
        if saved_state_path != None:

            state = torch.load(saved_state_path, map_location=torch.device(device))
            model.load_state_dict(state)

        self.model = model

        # initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, X_train, X_test, epochs):
        """Train netrade model with custom data
        
        Args:
            X_train : torch data loader -> data loader for training
            X_test : torch data loader -> data loader for testing
            epochs : int -> how many times do you want to train this model
        
        Returns:
            model : torch nn -> model that has been trained with your custom data
            history : dict -> accuracy, val_accuracy, loss, val_loss
        """
        history = {
            "accuracy": [],
            "val_accuracy": [],
            "loss": [],
            "val_loss": [],
        }

        # initialize accuracy score helper
        accuracy_score = Accuracy().to(device)

        for epoch in range(0, epochs):

            train_acc = 0
            train_loss = 0

            for X_chart, X_candle, train_label in X_train:

                # make all variable in same device
                X_chart = X_chart.to(device)
                X_candle = X_candle.to(device)
                train_label = train_label.to(device)

                # do the forward pass
                y_logit = self.model(X_chart, X_candle)
                train_preds = torch.softmax(y_logit, dim=1).argmax(1)

                # calculate accuracy
                acc = accuracy_score(train_preds, train_label)
                train_acc += acc

                # calculate the loss
                loss = self.loss_fn(y_logit, train_label)
                train_loss += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # inference mode
                self.model.eval()

                test_loss, test_acc = 0,0
                if X_test != None:

                    with torch.inference_mode():

                        for Test_chart, Test_candle, Test_label in X_test:

                            # make all variable to the same device
                            Test_chart = Test_chart.to(device)
                            Test_candle = Test_candle.to(device)
                            Test_label = Test_label.to(device)

                            test_logit = self.model(Test_chart, Test_candle)
                            test_preds = torch.softmax(test_logit, dim=1).argmax(1)

                            # calculate accuracy
                            acc_ = accuracy_score(test_preds, Test_label)
                            test_acc += acc_

                            # calculate the loss
                            loss_ = self.loss_fn(test_logit, Test_label)
                            test_loss += loss_

                        test_loss /= len(X_test)
                        test_acc /= len(X_test)

            train_loss /= len(X_train)
            train_acc /= len(X_train)

            # append data
            history['loss'].append(train_loss.detach().cpu().numpy())
            history['val_loss'].append(test_loss.detach().cpu().numpy())
            history['accuracy'].append(train_acc.detach().cpu().numpy())
            history['val_accuracy'].append(test_acc.detach().cpu().numpy())

            print(f'epoch : {epoch + 1} | loss : {train_loss:.5f} | val_loss : {test_loss:.5f} | accuracy : {train_acc:.5f} | val_accuracy : {test_acc:.5f}')

        # return the result
        return self.model, history

    def predict(self, chart_image, candle_image):
        """Predict price will go up or go down from single image
        
        Args:
            chart_image : torch tensor -> chart image that has been turned into tensor
            candle_image : torch tensor -> candlestick image that has been turned into tensor
        """

        with torch.inference_mode():

            logit = self.model(chart_image.unsqueeze(0).to(device), candle_image.unsqueeze(0).to(device))
            preds = torch.softmax(logit, dim=1)

        return preds