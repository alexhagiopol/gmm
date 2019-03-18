## Gaussian Mixture Models Tutorial

### Installation

    git clone https://github.com/alexhagiopol/gmm
    cd gmm

### Usage

At present the program only supports grayscale images. Parameter definitions:

    python3 gmm_segmentation.py image_filepath num_components num_iterations

Specific example:

    python3 gmm_segmentation.py example_data/beyonce.jpg 3 8

Example results:
    
![example_results](example_data/example_results.png)
