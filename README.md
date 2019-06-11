# stylize-datasets
This repository contains code for stylizing arbitrary image datasets using [AdaIN](https://arxiv.org/abs/1703.06868). The code is a generalization of Robert Geirhos' [Stylized-ImageNet](https://github.com/rgeirhos/Stylized-ImageNet) code, which is tailored to stylizing ImageNet. Everything in this repository is based on naoto0804's [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) implementation.
This repository is adapted from [Hvitgar/stylize-datasets](https://github.com/Hvitgar/stylize-datasets). Comparing with the original version, this adaption is
- 30% faster when processing a standardized data set; 
- compatible with PyTorch 1.0;
- more flexible.

Given an image dataset, the script creates the specified number of stylized versions of every image while keeping the directory structure and naming scheme intact (useful for existing data loaders or if directory names include class annotations).

Feel free to open an issue in case there is any question.

## Usage
- Download and convert the models:
    - ***Update*** Coverted `.pth` models are available in `models/`. No need to do this step any more. Loading `.t7` models is not supported in pytorch 1.0.
    - for this step it is mandatory that your pytorch version is 0.4.0. Run `pip install torch==0.4.0` (preferrably in a container/virtual environment in order to not mess with your local pytorch installation)
    - run `bash download_convert_models.sh`
- Get style images: Download train.zip from [Kaggle's painter-by-numbers dataset](https://www.kaggle.com/c/painter-by-numbers/data)
- To stylize a dataset, run `python stylize.py`.

    Arguments:
    - `--content-dir <CONTENT>` the top-level directory of the content image dataset (mandatory)
    - `--style-dir <STLYE>` the top-level directory of the style images (mandatory)
    - `--output-dir <OUTPUT>` the directory where the stylized dataset will be stored (optional, default: `output/`)
    - `--num-styles <N>` number of stylizations to create for each content image (optional, default: `1`)
    - `--alpha <A>` Weight that controls the strength of stylization, should be between 0 and 1 (optional, default: `1`)
    - `--extensions <EX0> <EX1> ...` list of image extensions to scan style and content directory for (optional, default: `png, jpeg, jpg`). Note: this is case sensitive, `--extensions jpg` will not scan for files ending on `.JPG`. Image types must be compatible with PIL's `Image.open()` ([Documentation](https://pillow.readthedocs.io/en/5.1.x/handbook/image-file-formats.html))
    - `--content-size <N>` Minimum size for content images, resulting in scaling of the shorter side of the content image to `N` (optional, default: `0`). Set this to 0 to keep the original image dimensions.
    - `--style-size <N>` Minimum size for style images, resulting in scaling of the shorter side of the style image to `N` (optional, default: `512`). Set this to 0 to keep the original image dimensions (for large style images, this will result in high (GPU) memory consumption).
    - `--crop` If set, content and style images will be cropped at the center to create square output images (optional)
    - `--batch-size <N>` Number of images to process simultaneously (greatly speeding up). Not working unless `--content-size` , `--style-size` are set to **non-zero** values, and `--crop` is enabled. (optional, default: `1`)
    - `--seed` Set the random seed (optional).

Examples:

- Keep the original image dimensions
```
 python3 stylize.py --content-dir '/home/stylize-datasets/images/' --style-dir '/home/stylize-datasets/train/' --num-styles 10 --content-size 0 --style-size 256
```
- Parallel processing
```
 python3 stylize.py --content-dir '/home/stylize-datasets/images/' --style-dir '/home/stylize-datasets/train/' --num-styles 10 --content-size 256 --style-size 128 --crop --batch-size 16 --seed 1234
```
