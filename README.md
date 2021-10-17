# U-Net with Optimal Depth and Width for Abdominal Organ Segmentation

Built upon [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet), this repository provides the solution of team LetsGo for FLARE21 Challenge.

## Environments and Requirements:
1. Install nnU-Net [1] as below. You should meet the requirements of nnUNet, our method does not need any additional requirements.  For more details, please refer to https://github.com/MIC-DKFZ/nnUNet
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

2. Set environment variables for nnU-Net. Concretely, Set the paths in your .bashrc file, which is located in your home directory. Open the file and add the following lines to the bottom:
```
export nnUNet_raw_data_base="/data/hzy/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/data/hzy/nnUNet/nnUNet_preprocessed"
export RESULTS_FOLDER="/data/hzy/nnUNet/nnUNet_trained_models"
```
(of course adapt the paths to your system)

3. Copy the python files in this repository to the code directory of nnUNet.
```
cp LetsGoTrainer.py nnunet/training/network_training
cp LetsGo_UNet.py nnunet/network_architecture
```

## Dataset
Download the training images, training labels and validation images from https://flare.grand-challenge.org/Data/.
Then organize the data of FLARE folowing the requirement of nnUNet.

    nnUNet_raw_data_base/nnUNet_raw_data/Task817_FLARE/
    ├── dataset.json
    ├── imagesTr
    ├── imagesTs
    └── labelsTr

## Preprocessing
 Conduct automatic preprocessing using nnUNet.
 ```
 nnUNet_plan_and_preprocessing -t 817
 ```

## Training
To train the model of our solution from scratch, run the following scripts:
```
cd nnunet/run
python run_training.py 3d_fullres LetsGoTrainer 817 all
```

## Trained Models
Download our trained model in [Baidu Net disk](https://pan.baidu.com/s/1SW7b-LJB6P1FM8mT4dZMSQ) (PW: orhp) and put the model in your RESULTS_FOLDER of nnUNet.

## Inference
```
python inference/predict_simple.py -i INPUT_FOLDER -o OUTPUT_FOLDER -t 817 -tr LetsGoTrainer -m 3d_fullres -f all --disable_tta
```

## Results
Our method achieves the following performance on the validation set of FLARE Challenge


| Metrics (Avg±Std) | nnUNet Baseline | LetsGo     |
| ------------------ | --------------- | ---------- |
| Liver-DSC          | 94.5±8.09      | 95.0±6.38 |
| Liver-NSD          | 79.3±14.9      | 80.3±14.8 |
| Kidney-DSC         | 80.4±17.0      | 80.0±18.3 |
| Kidney-NSD         | 70.9±18.4      | 71.3±18.7 |
| Spleen-DSC         | 89.5±18.0      | 90.6±16.7 |
| Spleen-NSD         | 82.0±19.3      | 83.9±19.6 |
| Pancreas-DSC       | 60.1±23.1      | 61.7±23.0 |
| Pancreas-NSD       | 50.6±17.7      | 51.5±18.8 |
| Running Time       | 145            | 188.1     |
| GPU Memory         | 2298           | 2938      | 

## Docker trained on AbdomenCT-1k
We retrained our model on the whole AbodomenCT-1K [2] dataset, and built a docker image of our trained model. You can download our model at [Baidu Net disk](https://pan.baidu.com/s/18u45XC04RyzkoQdeIMEtHw) (PW: 2021)
The docker can be used by running,:
```
docker image load < letsgo.tar.gz
docker container run --gpus "all" --name letsgo --rm \
-v $PWD/inputs/:/workspace/inputs/ \
-v $PWD/outputs/:/workspace/outputs/ \
letsgo:latest /bin/bash -c "sh predict.sh"
```

## Reference
[1] Isensee, Fabian, et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature methods 18.2 (2021): 203-211.

[2] Ma, Jun, et al. "Abdomenct-1k: Is abdominal organ segmentation a solved problem." IEEE Transactions on Pattern Analysis and Machine Intelligence (2021).
