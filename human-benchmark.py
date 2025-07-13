# for establishing benchmark rat-toad distribution 
# via inference on the UTKFace dataset

import os
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.io import decode_image

preload_distribution = True
distribution_path = "rat-toad.csv"

# for calculating a candidate image's position
candidate_face_path = "Chartwell"

# for calculating the distribution
path = "finetuned-15ep.pth"
reference_face_path = "UTKFace"
n_samples = 2000

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# load model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(path, weights_only=True))
model.to(device)

# input transformation for resnet18
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# run inference on random sample of the images if we're generating distribution
# otherwise run on our candidate faces and plot
# [1,0] rat [0,1] toad
model.eval()

with torch.no_grad():
    target = candidate_face_path if preload_distribution else reference_face_path
    file_list = os.listdir(target) if preload_distribution else random.sample(os.listdir(target), n_samples)
    images = []
    for fn in file_list:
        img_path = os.path.join(target, fn)
        image = decode_image(img_path, mode="RGB").float()
        image = transform(image).to(device)
        images.append(image)
    image_tensor = torch.stack(images, dim=0)
    print("Images loaded.")
    res = model(image_tensor)
    print("Inference completed.")
    
if preload_distribution:
    # load the distribution and compare it to the files we ran inference on
    df_reference = pd.read_csv(distribution_path)
    mean_rat = np.mean(df_reference["rat"])
    mean_toad = np.mean(df_reference["toad"])
    df_candidates = pd.DataFrame(res.cpu().numpy(), columns=["rat", "toad"])
    plt.scatter(df_reference["rat"], df_reference["toad"], c="lightgray", label="reference faces")
    plt.axvline(mean_rat, c="b")
    plt.axhline(mean_toad, c="b")
    plt.scatter(df_candidates["rat"], df_candidates["toad"], c="r", label="candidate faces")
    for i, fn in enumerate(file_list):
        plt.text(df_candidates["rat"][i]+0.05, df_candidates["toad"][i]-0.05, fn)
    plt.xlabel("Rat")
    plt.ylabel("Toad")
    plt.title("Candidate Face Rat-Toad Characteristics vs. References")
    plt.legend()
    plt.show()
else:
    # store the distribution
    df = pd.DataFrame(res.cpu().numpy(), columns=["rat", "toad"])
    df.to_csv(distribution_path, index=False)
    print("File written.")
