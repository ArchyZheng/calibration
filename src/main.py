from lightning.pytorch.loggers import CometLogger
from utils.read_data import read_data
from src.fitting_surface import FittingSurface, get_the_point_set_in_the_ellipse
import pytorch_lightning as pl
import torch
from src.TrainModule import TrainModule


def main():
    from torch.utils.data import DataLoader
    from src.fitting_surface import Polynomial
    file_name = '../data/pos_2.bin'
    image_data = read_data(file_name=file_name, width=512, height=512, read_type='double')
    fitting_process = FittingSurface(image_data=image_data)

    candidate_point = get_the_point_set_in_the_ellipse(ellipse_center=(236, 240), ellipse_axes=(369 // 2, 362 // 2),
                                                       ellipse_angle=109.59, original_map=(512, 512))
    training_set, validation_set = fitting_process.split_to_train_and_validation_set(candidate_set=candidate_point,
                                                                                     proportion=(0.8, 0.2))
    train_dataloader = DataLoader(dataset=training_set, batch_size=1, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_set, batch_size=1, shuffle=False)

    comet_logger = CometLogger(
        api_key='lNyK4LLQynW9EQrhnWPWfvHTk',
        project_name="OCT_new",
        experiment_name="MLP model_new"
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(logger=[comet_logger],
                         max_epochs=10,
                         accelerator=device)
    criterion = torch.nn.L1Loss()
    train_module = TrainModule(loss=criterion)

    trainer.fit(model=train_module, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
    comet_logger.experiment.end()


if __name__ == "__main__":
    main()
