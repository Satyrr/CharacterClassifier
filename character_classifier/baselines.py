from character_classifier.src.datasets.datamodule import CharactersDataModule
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

if __name__ == '__main__':
    """Fit and test baseline algorithms: Logistic regression and SVM"""
    data_module = CharactersDataModule()
    data_module.prepare_data()
    data_module.setup()

    train_data = list(data_module.train_dataloader())
    valid_data = list(data_module.val_dataloader())
    test_data = list(data_module.test_dataloader())

    train_x = np.vstack([x.numpy() for (x, _) in train_data])
    train_x = train_x.reshape(train_x.shape[0], -1)
    train_y = np.hstack([y.numpy().flatten() for (_, y) in train_data])

    valid_x = np.vstack([x.numpy() for (x, _) in valid_data])
    valid_x = valid_x.reshape(valid_x.shape[0], -1)
    valid_y = np.hstack([y.numpy().flatten() for (_, y) in valid_data])

    train_x = np.vstack([train_x, valid_x])
    train_y = np.hstack([train_y, valid_y])

    test_x = np.vstack([x.numpy() for (x, _) in test_data])
    test_x = test_x.reshape(test_x.shape[0], -1)
    test_y = np.hstack([y.numpy().flatten() for (_, y) in test_data])

    clf = LogisticRegression(random_state=0, max_iter=10)
    clf.fit(train_x, train_y)
    predictions = clf.predict(test_x)

    results = classification_report(test_y, predictions, output_dict=True)

    print('Logistic regression results')
    print('Accuracy = {}'.format(results['accuracy']))
    print('Macro f1 = {}'.format(results['macro avg']['f1-score']))

    clf = SVC(random_state=0, max_iter=10)
    clf.fit(train_x, train_y)
    predictions = clf.predict(test_x)

    results = classification_report(test_y, predictions, output_dict=True)

    print('SVM results')
    print('Accuracy = {}'.format(results['accuracy']))
    print('Macro f1 = {}'.format(results['macro avg']['f1-score']))