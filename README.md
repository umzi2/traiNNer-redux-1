# traiNNer-redux
## Overview
A modern community fork of [BasicSR](https://github.com/XPixelGroup/BasicSR) and [traiNNer-redux](https://github.com/joeyballentine/traiNNer-redux).

## Usage Instructions

### Initial Setup
1. Install [Python](https://www.python.org/) if it's not already installed. A minimum of version Python 3.11 is required. 
2. Install the [latest stable version of PyTorch](https://pytorch.org/get-started/locally/).
3. Clone the repository:
   - To use the git command line, navigate to where you want to install traiNNer-redux, and enter this command (install [git](https://git-scm.com/) first if it's not already installed):
      ```
      git clone https://github.com/the-database/traiNNer-redux.git
      ```
   - To use a GUI for git, follow the instructions for that git client. For [GitHub Desktop](https://desktop.github.com/), for example, click on the green Code button near the top of this page, click Open with GitHub Desktop and follow the instructions.
4. Open a command window in the `traiNNer-redux` directory where you cloned the repository. Run this command to install the remaining Python dependencies:
   ```
   pip install -r requirements.txt
   ```
### Training a Model
For more detailed information, refer to the wiki (under construction). The most commonly used settings are documented here. 

#### Set up config file
1. Navigate to `traiNNer-redux/options/train`, select the architecture you want to train, and open the `yml` file in that folder in a text editor. A text editor that supports YAML syntax highlighting is recommended, such as [VS Code](https://code.visualstudio.com/) or [Notepad++](https://notepad-plus-plus.org/). For example, to train SPAN, open `traiNNer-redux/options/train/SPAN/SPAN.yml`.
2. At the top of the file, set the `name` to the name of the model you want to train. Give it a unique name so you can differentiate it from other training runs.
3. Set the scale depending on what scale you want to train the model on. 2x doubles the width and height of the image, for example. Not all architectures support all scales. Supported scales appear next to the scale in a comment, so `# 2, 4` means the architecture only supports a scale of 2 or 4.
4. Set the paths to your dataset HR and LR images, at `dataroot_gt` and `dataroot_lq` under the `train:` section.
5. If you want to enable validation metrics, uncomment the validation sections under the set the paths to your validation HR and LR images, at `dataroot_gt` and `dataroot_lq` under the `val` section.
6. If you want to use a pretrain model, set the path of the pretrain model at `pretrain_network_g` and remove the `#` to uncomment that line.

#### Run command to start training

Run the following command to start training. Change `./options/train/arch/config.yml` to point to the config file you set up in the previous step. 
```
python train.py --auto_resume -opt ./options/train/arch/config.yml
```

For example, to train with the SPAN config: 
```
python train.py --auto_resume -opt ./options/train/SPAN/SPAN.yml
```

To pause training, press `ctrl+C` or close the command window. To resume training, run the same command that was used to start training. The `--auto_resume` flag will resume training from when it was paused. 

#### Test models

Models are saved in `traiNNer-redux/experiments/<name>/models`, where `name` is whatever was used in the config file. [chaiNNer](https://github.com/chaiNNer-org/chaiNNer) can be used to run the model. If you want to run the model on images during training to monitor the progress of the model, set up validation in the config file, and find the validation results in `traiNNer-redux/experiments/<name>/visualization`.

## Resources
- [OpenModelDB](https://openmodeldb.info/): Repository of AI upscaling models, which can be used as pretrain models to train new models. Models trained with this repo can be submitted to OMDB. 
- [chaiNNer](https://github.com/chaiNNer-org/chaiNNer): General purpose tool for AI upscaling and image processing, models trained with this repo can be run on chaiNNer. chaiNNer can also assist with dataset preparation.
- [WTP Dataset Destroyer](https://github.com/umzi2/wtp_dataset_destroyer): Tool to degrade high quality images, which can be used to prepare the low quality images for the training dataset.
- [helpful-scripts](https://github.com/Kim2091/helpful-scripts): Collection of scripts written to improve experience training AI models.
- [Enhance Everything! Discord Server](https://discord.gg/cpAUpDK): Get help training a model, share upscaling results, submit your trained models, and more. 

## License and Acknowledgement

This project is released under the [Apache 2.0 license](LICENSE.txt). More details about **license** and **acknowledgement** are in [LICENSE](LICENSE/README.md).

- This repository is a fork of [traiNNer-redux](https://github.com/joeyballentine/traiNNer-redux) which itself is a fork of [BasicSR](https://github.com/XPixelGroup/BasicSR).
- Network architectures are imported from [Spandrel](https://github.com/chaiNNer-org/spandrel).
- The SPANPlus architecture is from [umzi2/SPANPlus](https://github.com/umzi2/SPANPlus) which is a modification of [SPAN](https://github.com/hongyuanyu/SPAN).
- Several enhancements reference implementations from [Corpsecreate/neosr](https://github.com/Corpsecreate/neosr) and its original repo [neosr](https://github.com/muslll/neosr).
- Members of the Enhance Everything Discord server: [Corpsecreate](https://github.com/Corpsecreate), [joeyballentine](https://github.com/joeyballentine), [Kim2091](https://github.com/Kim2091).
