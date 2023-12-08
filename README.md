# CVPDL Final Project Baseline ResNet50 Model
## How to run this code
1. Clone this repository
2. Follow the instructions below to prepare your data & give correct commands

## Data Preparation
- Put all your data in the following manner
    - `classname` indicates the data label
    - Only `train` & `test` should not be changed, other pathname can be arbitrary name
```
data_root_directory/
    |- train
        |- classname_1
            |- img_1.jpg
            |- img_2.png
        |- classname_2
            |- img_1.jpg
            |- img_2.png
        .
        .
        .
    |- test (should be the same format as train/ folder)
        .
        .
        .
```

## Training
```
python main.py --train --device cuda --input_image_dir {data_root_directory} --model_path {resume_from_specific_checkpoint_if_needed}
```
- Models will be saved in `models/` folder by default, can be changed by `--output_model_dir {desired_folder}`
- Models are saved every **5** epochs by default, can be changed by `--save_every {desired_interval}`
- Training logs are saved in `./run_log.txt` by default, can be changed by `--log_path {desired_path}`

## Testing
```
python main.py --test --device cuda --input_image_dir {data_root_directory} --model_path {checkpoint_path_for_testing}
```