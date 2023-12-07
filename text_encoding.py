#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[ ]:


import torch
import torch.nn as nn
import requests
from PIL import Image
from transformers import AutoTokenizer, BlipProcessor, Blip2ForConditionalGeneration


# ### Tokenizing text using BLIP2 Tokenizer

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-flan-t5-xl")


# In[ ]:


class ReshapeText(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, tokenizer, token_dim=256, img_size=32):
      super().__init__()
      self.tokenizer = tokenizer
      self.token_dim = token_dim
      self.img_size = img_size
      self.resize = nn.Linear(self.token_dim, self.img_size**2)

    def forward(self, x):
      x = self.tokenizer(x, padding = 'max_length', truncation = True, max_length = 256)['input_ids']
      x = torch.tensor(x).float()
      return x


# In[ ]:


reshape_text = ReshapeText(tokenizer)

