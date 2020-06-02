import albumentations as albu


def get_training_augmentation():
    train_transform = [albu.HorizontalFlip(p=1)]
    return albu.Compose(train_transform)

def visualize(**images):
    import matplotlib.pyplot as plt
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
                    
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    #plt.show()
    plt.savefig('../outputs/test_aug/all.jpg')

def test01(aug):
    import glob
    import cv2
    import numpy as np
    image_paths = glob.glob('../inputs/train_data01/RAD_185196432784901/*.png')
    assert len(image_paths) == 41
    image_paths = sorted(image_paths)
    image_path = image_paths[0]
    img = cv2.imread(image_path)
    mask = np.where(img > 10,250,0) + np.where(img > 20,1,0) + np.where(img > 30,1,0)
    # img = np.transpose(img, (2,0,1))
    sample = aug(image=img, mask=mask) 
    img_aug,mask_aug = sample['image'], sample['mask']
    # cv2.imwrite('../outputs/test_aug/raw.jpg', img)
    # cv2.imwrite('../outputs/test_aug/aug.jpg', img_aug)
    # img_aug = np.transpose(img_aug, (1,2,0))
    visualize(image=img_aug, cars_mask=mask_aug.squeeze(),)

if __name__ == "__main__":
    aug = get_training_augmentation()
    test01(aug)
