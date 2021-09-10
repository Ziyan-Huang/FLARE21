# U-Net with Optimal Depth and Width for Abdominal Organ Segmentation

Built upon [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet), this repository provides the solution of team LetsGo for FLARE21 Challenge.

## Environments and Requirements:
1. Install nnU-Net as below. You should meet the requirements of nnUNet, our method does not need any additional requirements.  For more details, please refer to https://github.com/MIC-DKFZ/nnUNet
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

2. Set environment variables for nnU-Net. Concretely, Set the paths in your .bashrc file, which is located in your home directory. Open the file and add the following lines to the bottom:
```
export nnUNet_raw_data_base="/media/fabian/nnUNet_raw"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export RESULTS_FOLDER="/media/fabian/nnUNet_trained_models"
```
(of course adapt the paths to your system)

3. Copy the python files in this repository to the code directory of nnU-Net.
* Copy the python files in folder network_architecture to nnunet/architecture
* Copy the python files in folder network_training to nnunet/training/network_training
* Copy the python file run_searching.py to nnunet/run
## Dataset
FLARE21
We name the dataset Task817_FLARE.

## Preprocessing
 Conduct automatic preprocessing using nnUNet.
 ```
 nnUNet_plan_and_preprocessing -t 817
 ```

## Training
To train the model of our solution, run the following scripts:
```
cd nnunet/run
python run_training.py 3d_fullres LetsGoTrainer 817 all
```

## Trained Models
We will release our trained model later.

## Inference
```
python inference/predict_simple.py -i INPUT_FOLDER -o OUTPUT_FOLDER -t 817 -m 3d_fullres -f all --disabled_tta
```

## Results
Our method achieves the following performance on the validation set of FLARE Challenge


| Metrics (Avg+-Std) | nnUNet Baseline | LetsGo     |
| ------------------ | --------------- | ---------- |
| Liver-DSC          | 94.5±8.09      | 95.0±6.38 |
| Liver-NSD          | 79.3±14.9      | 80.3±14.8 |
| Kidnet-DSC         | 80.4±17.0      | 80.0±18.3 |
| Kidney-NSD         | 70.9±18.4      | 71.3±18.7 |
| Spleen-DSC         | 89.5±18.0      | 90.6±16.7 |
| Spleen-NSD         | 82.0±19.3      | 83.9±19.6 |
| Pancreas-DSC       | 60.1±23.1      | 61.7±23.0 |
| Running Time       | 145             | 188.1      |
| GPU Memory         | 2298            | 2938       | 


