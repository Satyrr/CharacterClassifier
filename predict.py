import numpy as np
import torch
import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader

from character_classifier.src.datasets.datamodule import CharactersDataModule
from character_classifier.src.models.classifier import CharacterClassifier


def predict(input_data):
    clf = CharacterClassifier.load_from_checkpoint('./character_classifier/data/best_models/eff0.ckpt')
    clf.eval()

    input_data = torch.tensor(input_data).float()
    input_data = input_data.view(input_data.size(0), 1, 56, 56)

    outputs = []
    with torch.no_grad():
        for x in tqdm.tqdm(DataLoader(TensorDataset(input_data), batch_size=32),
                           total=len(input_data) / 32):
            predictions = clf(x[0]).cpu().numpy().argmax(axis=1)
            outputs.append(predictions)

    return np.hstack(outputs)


if __name__ == '__main__':
    data_module = CharactersDataModule()
    data_module.prepare_data()
    data_module.setup()

    test_data = list(data_module.test_dataloader())

    test_x = np.vstack([x.numpy() for (x, _) in test_data])
    test_y = np.hstack([y.numpy().flatten() for (_, y) in test_data])

    predictions = predict(test_x)

    print(classification_report(test_y.flatten(), predictions))
