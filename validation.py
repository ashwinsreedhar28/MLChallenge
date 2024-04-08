import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model_utility import VisionTransformerClassifier  
from pathlib import Path

def load_trained_network(network_path, class_total):
    network = VisionTransformerClassifier(class_count=class_total)
    network.load_state_dict(torch.load(network_path))
    network.eval() 
    return network

def setup_data_loader(image_directory, image_transforms):
    inference_set = ImagePredictionDataset(root_path=image_directory, image_transform=image_transforms)
    loader = DataLoader(inference_set, batch_size=32, shuffle=False, num_workers=4)
    return loader

class ImagePredictionDataset(Dataset):
    def __init__(self, root_path, image_transform=None):
        self.root_path = Path(root_path)
        self.image_transform = image_transform
        self.image_files = sorted(self.root_path.glob('*'), key=lambda x: int(x.stem))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.image_transform:
            image = self.image_transform(image)

        return image, image_path.stem

def predict_categories(image_loader, model, device, category_labels):
    predictions, file_ids = [], []
    with torch.no_grad():
        for images, ids in image_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            file_ids.extend(ids)

    predicted_categories = [category_labels.iloc[p]['Category'] for p in predictions]
    
    return pd.DataFrame({'Id': file_ids, 'Category': predicted_categories})

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    network = load_trained_network('transformer_model.pth', 100).to(device)

    category_labels = pd.read_csv('category.csv')

    loader = setup_data_loader('test', image_transforms)
    prediction_results = predict_categories(loader, network, device, category_labels)

    prediction_results.to_csv('inference_results.csv', index=False)

    print("Prediction complete and results saved.")

if __name__ == "__main__":
    main()
