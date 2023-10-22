import streamlit as st
import torchvision
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

def load():
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

    test = torchvision.datasets.MNIST(
        root="data",
        train=False,
        download=False,
        transform=transform
    )
    return test

def show(test):
    sample_idx = torch.randint(len(test), size=(1,)).item()
    st.write(f"Index gambar : {sample_idx}")
    image, label = test[sample_idx]
    # out = plt.imshow(image.squeeze(), cmap = 'gray')
    img = transforms.ToPILImage()(image)
    return img, label



st.title("Resnet with MNIST")

test = load()
img, label = show(test)
upscale = transforms.Compose([transforms.Resize((224,224))])
st.image(upscale(img))
st.write(f"Label dari gambar tersebut adalah {label}")