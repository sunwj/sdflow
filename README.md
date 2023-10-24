# SDFlow

Pytorch implementation of paper **"Learning Many-to-Many Mapping for Unpaired Real-World Image Super-resolution and Downscaling"**


### Pre-trained models
You can download the pre-trained models from [here](https://mega.nz/folder/f7QWBLqA#bAijxQ2iGF1PsSgK1Qm38w).

## Inference
	python3 sdflow_sr.py --lr_imgs_path path_to_lr_images --output_path path_for_sr_images --pretrained_model_path path_to_pretrained_models --tau sampling_temperature
	python3 sdflow_ds.py --hr_imgs_path path_to_hr_images --output_path path_for_downscaled_images --pretrained_model_path path_to_pretrained_models --tau sampling_temperature


If you find our work useful in your research or publication, please cite our work:

Wanjie Sun, Zhenzhong Chen. **"Learning Many-to-Many Mapping for Unpaired Real-World Image Super-resolution and Downscaling"**. arXiv preprint arXiv:2310.04964, 2023.

## Full code include training scripts will be available later.
