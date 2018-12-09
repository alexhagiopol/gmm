## Gaussian Mixture Models Tutorial

### Installation

    git clone https://github.com/alexhagiopol/gmm
    cd gmm

### Usage

At present the program only supports grayscale images. Parameter definitions:

    python gmm_segmentation.py image_filepath num_components num_iterations

Specific example:

    python gmm_segmentation.py images/beyonce.jpg 3 8

Example results:
    
![example_results](images/example_results.png)
