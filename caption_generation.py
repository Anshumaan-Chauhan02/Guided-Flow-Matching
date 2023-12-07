#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[ ]:


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import requests
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration
import matplotlib.pyplot as plt
import numpy as np
import pickle


# ### Helper Functions

# In[ ]:


def imshow(img):
    img = img
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# ### Loading the Caption Generation Model

# In[ ]:


def create_processor_and_model():
  processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
  model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16)

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)
  return processor, model


# ### Loading the Train and Test CIFAR Dataset

# In[ ]:


def get_dataset_and_loader():
  transform = transforms.Compose(
    [transforms.ToTensor()])

  batch_size = 4

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
  classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  return trainset, trainloader, testset, testloader


# ### Custom Dataset for the Images with their Captions

# In[ ]:


class CIFAR10WithCaptions(Dataset):
    def __init__(self, cifar_dataset, captions):
        self.cifar_dataset = cifar_dataset
        self.captions = captions


    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        caption = self.captions[idx]
        return image, label, caption


# ### Generating captions and creating a new pickle file for the dataset

# In[ ]:


def generate_caption_dataset(dataset, dataset_loader, processor, model):
  captions = []
  for images, labels in dataset_loader:
    for image in images:
         transform = transforms.ToPILImage()
         PIL_image = transform(image)

         question = "Describe everything in this image"
         inputs = processor(PIL_image, question, return_tensors="pt").to("cuda").to("cuda", torch.float16)

         out = model.generate(**inputs)
         captions.append(processor.decode(out[0], skip_special_tokens=True))
  return CIFAR10WithCaptions(dataset, captions)

