# **Image-based food recognition** with U-Nets 
The repository contains the project realized for the *Deep Learning* course of the [Master's degree in Artificial Intelligence](https://corsi.unibo.it/2cycle/artificial-intelligence), at Alma Mater Studiorum, University of Bologna.

The projects is realized by:
* Alex Costanzino ([arcanoXII](https://github.com/arcanoXIII)), alex.costanzino@studio.unibo.it;
* Marco Costante ([Markostante](https://github.com/Markostante)), marco.costante@studio.unibo.it.

## Contents
* `01_dimension_check.py` and `02_reordering_dataset.py` are just auxiliary tools to pre-process the dataset;
* `u_net_for_image-based_food_segmentation.ipynb` is the main notebook;
* `deep_u_net_for_image-based_food_segmentation.py` is the alternative architecture;
* `demo_evaluation.ipynb` is a notebook for a quick evaluation of the various models;
* `report.pdf` is the report file.

## Main libraries
* [TensorFlow](https://www.tensorflow.org/) with [Keras](https://keras.io/) backend;
* [Numpy](https://numpy.org/) 1.20.0;
* [Pandas](https://pandas.pydata.org/) 1.2.4;
* [OpenCV](https://opencv.org/) 4.5.2;
* [PIL](https://pillow.readthedocs.io/) 8.2.0;
* [pycocotools](https://github.com/cocodataset/cocoapi);
* [tqdm](https://tqdm.github.io/) 4.61.0;
* [matplotlib](https://matplotlib.org/) 3.4.2;
* [segmentation_models](https://segmentation-models.readthedocs.io/).

Further details can be found in the report.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for further details.
