import os
import pandas as pd
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from transformers import ViTForImageClassification, ViTConfig

# Custom Vision Transformer model for image classification
class VisionTransformerClassifier(nn.Module):
    def __init__(self, class_count: int):
        super(VisionTransformerClassifier, self).__init__()
        self.configuration = ViTConfig.from_pretrained('google/vit-base-patch16-224', num_labels=class_count)
        self.vision_transformer = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224', config=self.configuration, ignore_mismatched_sizes=True)

        if not isinstance(self.vision_transformer, ViTForImageClassification):
            raise TypeError("Model not supported for image classification")

    def forward(self, inputs):
        result = self.vision_transformer(inputs)
        return result.logits

class ImageLoader(Dataset):
    def __init__(self, img_csv, label_csv, data_dir, img_transform=None):
        self.img_labels = pd.read_csv(img_csv)
        self.label_map = pd.read_csv(label_csv)
        self.directory = data_dir
        self.img_transform = img_transform

    def __len__(self):
        return len(os.listdir(self.directory))

    def __getitem__(self, idx):
        img_name = os.listdir(self.directory)[idx]
        img_id = img_name.split('.')[0].split('_')[1]
        
        category_name = self.img_labels.iloc[int(img_id)]['Category']
        category_label = self.label_map.loc[self.label_map['Category'] == category_name].index[0]

        img_file_path = os.path.join(self.directory, img_name)
        img_data = Image.open(img_file_path)
        
        if self.img_transform:
            img_data = self.img_transform(img_data)

        return img_data, category_label

def execute_training():
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    face_data = ImageLoader('train.csv', 'category.csv', 'train-cropped', data_transforms)

    dataset_size = len(face_data)
    train_size = int(0.8 * dataset_size)
    validation_size = dataset_size - train_size

    train_data, validation_data = random_split(face_data, [train_size, validation_size])

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_data, batch_size=32, shuffle=False, num_workers=4)

    total_classes = 100
    vision_model = VisionTransformerClassifier(class_count=total_classes)

    loss_fn = nn.CrossEntropyLoss()
    optimizer_fn = optim.Adam(vision_model.parameters(), lr=0.001)

    compute_device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    vision_model.to(compute_device)

    training_epochs = 50

    for epoch in range(training_epochs):
        vision_model.train()
        for batch_index, (imgs, lbls) in enumerate(train_dataloader):
            imgs, lbls = imgs.to(compute_device), lbls.to(compute_device)
            
            optimizer_fn.zero_grad()
            predictions = vision_model(imgs)
            loss = loss_fn(predictions, lbls)
            loss.backward()
            optimizer_fn.step()

            if (batch_index + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{training_epochs}], Batch [{batch_index + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

        vision_model.eval()
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for imgs, lbls in validation_dataloader:
                imgs, lbls = imgs.to(compute_device), lbls.to(compute_device)
                predictions = vision_model(imgs)
                _, preds = torch.max(predictions, 1)
                total_preds += lbls.size(0)
                correct_preds += (preds == lbls).sum().item()

            print(f'Validation Accuracy: {100 * correct_preds / total_preds:.2f}%')

    torch.save(vision_model.state_dict(), 'transformer_model.pth')

if __name__ == '__main__':
    execute_training()
