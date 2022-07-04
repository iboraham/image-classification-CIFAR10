import albumentations


def get_train_transformations():
    return albumentations.Compose(
        [
            albumentations.Resize(256, 256),
            albumentations.RandomCrop(224, 224),
            albumentations.HorizontalFlip(),
            albumentations.VerticalFlip(),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_test_transformations():
    return albumentations.Compose(
        [
            albumentations.Resize(256, 256),
            albumentations.CenterCrop(224, 224),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
