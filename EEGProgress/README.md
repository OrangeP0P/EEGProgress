# EEGProgress

A fast and lightweight progressive convolutoin architecture for EEG processing and classification

## Paper link

This project is based on a recent publication. You can access the original paper here: [EEGProgress](https://www.sciencedirect.com/science/article/pii/S0010482523013665).

![Alt text](ReadMe/EEGProgress.png)

# How to run

You can directly run the code with 1-AB.py

This application is designed to run in a PyTorch environment. To execute `1-AB.py`, which is the entry point of the program, follow the steps outlined below.

## Prerequisites

Before running the application, ensure that you have the following prerequisites installed:

1. **Python:** The code is tested with Python 3.8. It should be compatible with most Python 3.x versions.

2. **PyTorch:** This project requires PyTorch. If you haven't installed PyTorch yet, you can find installation instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).


## Running 1-AB.py

Once you have the environment set up, you can run `1-AB.py` by following these steps:

1. Open your command line interface (CLI).

2. Navigate to the directory where `1-AB.py` is located.

3. Select the network. You can select the correspoding testing model such as 'EEGProgress' with the setting in '1-AB.py':

   ```bash
    Net_number = 'EEGProgress' # Choose EEGProgress model
    ```
   
5. Run the script with the command:

    ```bash
    python 1-AB.py
    ```
