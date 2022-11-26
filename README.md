## Gaussian Mixture Models Tutorial

## Introduction

This repository is the accompanying software for my mathematics and programming tutorial paper for Gaussian Mixture Models. See [https://alexhagiopol.github.io/content/gmm_tutorial.pdf](https://alexhagiopol.github.io/content/gmm_tutorial.pdf). 

### Installation
Below commands tested on Ubuntu 22.04: 

    git clone https://github.com/alexhagiopol/gmm
    cd gmm
    sudo apt install git
    sudo apt install pip3
    sudo apt install python3-tk
    sudo apt install python3-pil python3-pil.imagetk
    pip install -r requirements.txt

### Usage
Command line parameter definitions:

    -h, --help      Show help message.
    --first-image   Path to image file. Must be specified.
    --second-image  Path to image file. May or may not be specified.
    --components    Number of components in the mixture of Gaussians. Must be specified.
    --iterations    Number of Expectation Maximization iterations. Must be specified.

#### Example Commands
Segment a single image:

    python gmm_segmentation.py --first-image=example_data/beyonce.jpg --components=3 --iterations=8

Segment the difference between a pair of images (reproduce Figure 7 in paper):

    python gmm_segmentation.py --first-image=example_data/image_pairs/2_background.png --second-image=example_data/image_pairs/2_foreground.png --components=2 --iterations=6 --subtraction-threshold=5.0

### Example results:
Single image segmentation into 3 components (approximately "white", "black", and "grey") over 8 iterations:
    
![example_results](example_data/example_results.png)
