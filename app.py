import streamlit as st
import random
import torchvision
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from model import ResNet50


def load():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    test = torchvision.datasets.MNIST(
        root="data", train=False, download=False, transform=transform
    )
    return test


def dataloader(test):
    test_dataloader = DataLoader(test, batch_size=100, shuffle=True, num_workers=0)
    return test_dataloader


def show(input, label):
    upscale = transforms.Compose([transforms.Resize((224, 224))])
    idx = random.randrange(99)
    img = transforms.ToPILImage()(input[idx])
    st.image(upscale(img))
    st.write(f"Data yang saya gunakan berlabel {label[idx]}")
    return input, label[idx], idx


def pred(image, label, idx):
    model = ResNet50(img_channel=1, num_classes=10).to(device)
    PATH = "model"
    model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
    model.eval().to(device)
    out = model(image)
    pred = out.argmax(dim=1, keepdim=True)

    if pred[idx] == label:
        st.write("Prediksi benar")
        st.write(f"Model memprediksi {int(pred[idx])}")
    else:
        st.write("Prediksi salah")
        st.write(f"Model memprediksi {pred[idx]}")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
st.title("Resnet with MNIST")

test = load()
data = dataloader(test)
input, labels = next(iter(data))


if st.button("Generate and predict"):
    st.write("Data yang digunakan")
    image, label, idx = show(input, labels)
    pred(image, label, idx)
