import torch
from PIL import Image
from torch.utils.data import DataLoader
import os

class DefectDescDataset(torch.utils.data.Dataset):
    '''
    Dataset class for descriptions of the defective images
    '''
    def __init__(self, dataset, processor, prompt, img_directory):
        self.dataset = dataset
        self.processor = processor
        self.prompt = prompt
        self.img_directory = img_directory

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        answer = self.dataset.loc[idx]['output']
        image_id = self.dataset.loc[idx]['file_name']
        image_path = os.path.join(self.img_directory, image_id)
        image = Image.open(image_path).convert("RGB")
        
        encoding = self.processor(image, self.prompt, return_tensors="pt")
        labels = self.processor.tokenizer.encode(answer, return_tensors='pt')

        return encoding, labels
    
def get_dataloader(config, train_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                  shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    return train_dataloader