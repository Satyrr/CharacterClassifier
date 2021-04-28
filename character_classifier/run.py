from src.training.train import train
from src.models.models.cnn import SimpleCNN

if __name__ == '__main__':

    result = train(model_cls=SimpleCNN)

    print(result)