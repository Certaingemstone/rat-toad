# rat-toad
A viral TikTok trend once classified people as either "rat" or "frog" in appearance. This is a classifier to do so systematically, using ResNet18.

DISCLAIMER: This is by no means a scientific test. It is purely the result of a coworker's meme. Quality and relevance of training data was not verified, and there was no attempt at feature engineering to accurately identify things humans will recognize as rat or frog in facial features. Agreement with human judgment is therefore not guaranteed.

![plot](https://github.com/Certaingemstone/rat-toad/blob/main/rat-toad.png?raw=true)

# Dependencies
- Pytorch
- Pandas, Numpy, Matplotlib

# Usage from pretrained model and distribution
This repository already has finetuned model weights and a pre-calculated distribution of rat- and frog-like human faces. These are used in the `human-benchmark` script. To use, create a directory with the faces you want to classify, cropped similarly to the reference faces in the UTKFace *cropped* dataset. Set this as the `candidate_face_path` and set `preload_distribution` to True. Running the script should plot your faces' positions in the distribution.

# Usage from scratch
Create directories "Rats" and "Frogs" containing images of rats and frogs for training. I used separate datasets for [rats](https://www.kaggle.com/datasets/ojoolasehindeitunu/rodents) and for [frogs](https://github.com/jonshamir/frog-dataset). The `making-annotations` script produces a CSV annotating the images.

Copy all images into a directory "all" for training. The `trainer` script finetunes the model to classify rats and frogs.

Create a directory with reference human faces, ideally a broad dataset. The `human-benchmark` script will run inference on these references to estimate a typical population distribution of rat and frog. Modify the `reference_face_path` variable in the script to point to this directory, and set `preload_distribution` to False. Running the script should save the resulting distribution to a file.

Finally, move to the steps to "Usage from pretrained model and distribution" above.
