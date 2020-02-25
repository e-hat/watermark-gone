# watermark-gone
A simple watermark remover that puts random watermarks on images, then uses an autoencoder built with TensorFlow to learn to remove those watermarks to reconstruct the original image. It is not for illegal use and only exists to help me learn more.

## Acknowledgements
I did NOT write the file 'stl10_input.py'. I got that from the [STL10 website](http://ai.stanford.edu/~acoates/stl10/), and more specifically,  [this Github page](https://github.com/mttk/STL10/blob/master/stl10_input.py). Otherwise, I used stackoverflow numerous times to learn more about `numpy` and I'm sure I'll add to this section once I get into the actual ANN design.

## Requirements
[Python 3.7.6](https://www.python.org/downloads/release/python-376/)  
[SciPy](https://scipy.org/install.html)  
[Pillow](https://pillow.readthedocs.io/en/stable/installation.html)  

## Usage
For now, all you can do is load data. Before you run `main.py`, download the STL10 data with `python load_data.py`, or however you run python files on your system. Then, you will be able to `python main.py`.
