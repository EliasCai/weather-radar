import albumentations as albu


def get_training_augmentation():
    train_transform = [albu.HorizontalFlip(p=1)]
    return albu.Compose(train_transform)

