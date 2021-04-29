from character_classifier.src.training.train import train
from character_classifier.src.models.models.cnn import SimpleCNN
from character_classifier.src.models.models.efficientnet import EfficientNet
from functools import partial

if __name__ == '__main__':
    train(model_cls=SimpleCNN, num_epochs=3, lr=0.001, kernel_size_1=5, filters_num_2=50)

    #eff_net_cls = partial(EfficientNet, 'tf_efficientnet_b0_ns', True)
    #train(model_cls=eff_net_cls, num_epochs=10, lr=0.0001, batch_size=4)
