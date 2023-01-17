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

    python3 gmm_segmentation.py --first-image=example_data/beyonce.jpg --components=3 --iterations=8

Segment the difference between a pair of images (reproduce Figure 7 in paper):

    python3 gmm_segmentation.py --first-image=example_data/image_pairs/2_background.png --second-image=example_data/image_pairs/2_foreground.png --components=2 --iterations=6 --subtraction-threshold=5.0

Run the code with no visualization for profiling purposes:

    time python3 gmm_segmentation.py --first-image=example_data/beyonce.jpg --components=3 --iterations=8 --visualization=0

###### Using Precompiled C++ Functions
This project has custom-implemented C++ functions to increase performance which is especially noticeable on larger resolution datasets like the example image `church.jpg`. These functions require (a) submodules to be cloned, (b) existing installation of a C++ compiler invokable by `make` and CMake invokable by `cmake`, (c) running the precompilation script:

    git clone --recursive git@github.com:alexhagiopol/pybind_examples.git
    cd gmm
    python3 build_precompiled_functions.py

Once `build_precompiled_functions.py` finishes successfully, the GMM segmentation can be re-run by using the `--precompiled-num-threads` parameter to specify a max number of threads to use for the precompiled functions.

Run the code with no visualization for profiling purposes:

    time python3 gmm_segmentation.py --first-image=example_data/church.jpg --components=3 --iterations=8 --visualization=0 --precompiled-num-threads=12

### Example results:
Single image segmentation into 3 components (approximately "white", "black", and "grey") over 8 iterations:
    
![example_results](example_data/example_results.png)
