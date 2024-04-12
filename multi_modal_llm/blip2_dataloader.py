import torch
from PIL import Image
from torch.utils.data import DataLoader
import os

class DefectDescDataset(torch.utils.data.Dataset):
    '''
    Dataset class for descriptions of the defective images
    '''
    def __init__(self, dataset, processor, prompt, img_directory, max_padding_length = 77):
        self.dataset = dataset
        self.processor = processor
        self.prompt = prompt
        self.img_directory = img_directory
        self.max_padding_length = max_padding_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        answer = self.dataset.loc[idx]['output']
        image_id = self.dataset.loc[idx]['file_name']
        image_path = os.path.join(self.img_directory, image_id)
        image = Image.open(image_path).convert("RGB")
        
        encoding = self.processor(image, self.prompt, return_tensors="pt")
        labels = self.processor.tokenizer.encode(answer, return_tensors='pt').squeeze(0)

        return encoding, labels
    
def collate_fn(batch):
  '''
  As each label/answer has a different output size, we need to use collate_fn
  as custom dataloader must have same dimension for each element in the batch
  '''
  from torch.nn.utils.rnn import pad_sequence
  
  #Collating the metadata
  pixel_values = torch.stack([x[0]['pixel_values'] for x in batch])
  input_ids = torch.stack([x[0]['input_ids'] for x in batch])
  attention_mask = torch.stack([x[0]['attention_mask'] for x in batch])

  #Collating the labels
  labels = [x[1] for x in batch]
  padded_labels = pad_sequence(labels, batch_first=True)

  return pixel_values, input_ids, attention_mask, padded_labels

def get_dataloader(config, train_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                  shuffle=True, 
                                  #collate_fn=collate_fn,
                                  num_workers=config['num_workers'], pin_memory=True)
    return train_dataloader