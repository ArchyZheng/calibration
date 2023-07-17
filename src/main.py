from lightning.pytorch.loggers import CometLogger, WandbLogger
from utils.read_data import read_data
from src.fitting_surface import FittingSurface, get_the_point_set_in_the_ellipse
import pytorch_lightning as pl
import torch
from src.TrainModule import TrainModule
from torch.utils.data import DataLoader
from src.fitting_surface import Polynomial_1, Polynomial_2, Polynomial_3, MLP


def main():
    file_name = '../data/pos_2.bin'
    image_data = read_data(file_name=file_name, width=512, height=512, read_type='double')
    fitting_process = FittingSurface(image_data=image_data)

    candidate_point = get_the_point_set_in_the_ellipse(ellipse_center=(236, 240), ellipse_axes=(369 // 2, 362 // 2),
                                                       ellipse_angle=109.59, original_map=(512, 512))
    training_set, validation_set = fitting_process.split_to_train_and_validation_set(candidate_set=candidate_point,
                                                                                     proportion=(0.8, 0.2))
    train_dataloader = DataLoader(dataset=training_set, batch_size=1, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_set, batch_size=1, shuffle=False)
    wandb_logger = WandbLogger(project='Calibration', name='Polynomial 1')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(logger=[wandb_logger],
                         max_epochs=10,
                         accelerator=device)
    creteria = torch.nn.L1Loss()
    model = Polynomial_1()

    train_module = TrainModule(loss=creteria, model=model)
    # train_module.load_from_checkpoint(checkpoint_path='../src/trial.ckpt')

    trainer.fit(model=train_module, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
    wandb_logger.experiment.finish()
    trainer.save_checkpoint(filepath="trial.ckpt", weights_only=True)


if __name__ == "__main__":
    main()