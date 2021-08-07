import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "C:/Users/nirya/Datasets/horse2zebra/horse2zebra"
VAL_DIR = "C:/Users/nirya/Datasets/horse2zebra/horse2zebra"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10 # according to paper
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
CHECK_POINT_GEN_H = "genh.pth.tar" # horse gen' checkpoint
CHECK_POINT_GEN_Z = "genz.pth.tar" # zebra gen' checkpoint
CHECKPOINT_DISC_H = "disch.pth.tar"
CHECKPOINT_DISC_Z = "discz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0":"image"}, # for applying the same augmentation with the same parameters to multiple images, in our case, image0 and image
)
