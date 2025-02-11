from torchvision import transforms
 
 transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip horizontally
        transforms.RandomRotation(degrees=20), # Randomly rotate up to 20 degrees
        transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.2, hue=0.5),  # Adjust color
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([ # No augmentation on validation set
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])



def get_training_transforms_224():
    return transform_train

def get_validation_transforms_224():
    return transform_val