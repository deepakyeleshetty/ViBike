# ViBike: Road Quality Classification Model


## Usage

`python main.py --action fl --round <r> --clients <c> -- batch_size <b> --epochs <e> --name <any_name>`

Notes:
* A new directory will be created in "runs". The plots and the model parameters will be saved in this directory.
* For now there are only 2 clients (from RoadData folder)
* The main functions used are:
  * `main.py`--> `init(), do_fl(), do_local_mlps(), test_model_updates(), and test_loop()` 
  * `data.py`--> `read_client_data(), load_test_data()`. These functions are customized to this dataset.

## Description

The main files used in this code are:
* `main.py` does everything
  * Argument `--action` can be passed, with values `svm` or `mlp` or `fl`
* `data.py` to 
  * `read_data(mode)` function loads the csv files from the `Results` directory
  * The CSV file from each class is combined into a list and is returned
  * Binary SVM requires labels to be either 1 or -1, while MLP needs values >0. So change the label to 2 by modifying the variable `b`.
* `model.py` which consists of the classifiers
  * `MLP` - Multi Layer Perceptron
  * `SVM2` - Binary / 2-class SVM classifier
  * `MultiClassSVM` - SVM classifier for multiple classes. 
    * Computes the output in vectorized form for all the points in the dataset.
* `IMUDataLoader.py` consists of the custom Dataset class for MLP. It inherits PyTorch's Dataset class.

For now, the remaining `.py` files and `.ipynb` files can be ignored.



