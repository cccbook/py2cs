from pl_bolts.models.regression import LinearRegression
import pytorch_lightning as pl
from pl_bolts.datamodules import SklearnDataModule
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
loaders = SklearnDataModule(X, y)

model = LinearRegression(input_dim=10)
trainer = pl.Trainer()
trainer.fit(model, train_dataloaders=loaders.train_dataloader(), val_dataloaders=loaders.val_dataloader())
trainer.test(test_dataloaders=loaders.test_dataloader())