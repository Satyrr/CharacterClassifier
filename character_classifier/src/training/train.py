from typing import Any

import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from character_classifier.settings import STORAGE_DIR
from character_classifier.src.datasets.datamodule import CharactersDataModule
from character_classifier.src.models.classifier import CharacterClassifier


def train(model_cls: type, lr: float = 0.01, num_epochs: int = 3, batch_size: int = 64,
          **kwargs) -> Any:
    """Train classifier."""
    pl.seed_everything(42)

    data_module = CharactersDataModule(batch_size=batch_size)
    data_module.prepare_data()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    model = CharacterClassifier(model_cls=model_cls, lr=lr, **kwargs)

    logging_path = STORAGE_DIR / 'logs'
    checkpoint_path = STORAGE_DIR / 'models'

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_top_k=1,
        monitor='valid_loss',
        mode='min'
    )

    tb_logger = pl_loggers.TensorBoardLogger(logging_path)

    is_gpu_available = torch.cuda.is_available()
    gpu_number = 1 if is_gpu_available else 0

    trainer = pl.Trainer(gpus=gpu_number,
                         max_epochs=num_epochs,
                         progress_bar_refresh_rate=20,
                         checkpoint_callback=checkpoint_callback,
                         logger=tb_logger)

    trainer.fit(model, train_loader, val_loader)

    checkpoint = torch.load(
        checkpoint_callback.best_model_path, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])

    result = trainer.test(test_dataloaders=test_loader)

    return result
