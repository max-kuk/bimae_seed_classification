# Official TensorFlow implementation of the paper "BiMAE - A Bimodal Masked Autoencoder Architecture for Single-Label Hyperspectral Image Classification" (CVPRW 2024)

## Input Data
The input data should be in the form of TFRecords. The TFRecords should contain the following features:
- 'id': tf.string,
- 'rgb_image': tf.float32,
- 'hs_image': tf.uint8,
- 'label': tf.string


## Pretraining

<img src="imgs/BiMAE_pretraining.jpg" alt="BiMAE pretraining" style="width:600px;"/>
To train the model, run the following command:
```bash
nohup python mae_trainer.py --model=mae_vit_tiny_patch24 --scr_dir=path/to/tfrecord --batch_size=512 --epochs=300 --patch_size=24 --hs_image_size=24 --hs_num_patches=300 --hs_mask_proportion=0.9 --rgb_image_size=192 --rgb_num_patches=64 --hs_mask_proportion=0.75 > mae_trainer.log &  
```

## Finetuning


<img src="imgs/BiMAE_finetuning.jpg" alt="BiMAE finetuning" style="width:300px;"/>

To fine-tune the model, run the following command:
```bash
nohup python mae_trainer_finetuning.py --model=mae_vit_tiny_patch24 --scr_dir=path/to/tfrecord --batch_size=512 --epochs=50 --patch_size=24 --hs_image_size=24 --hs_num_patches=300 --hs_mask_proportion=0.9 --rgb_image_size=192 --rgb_num_patches=64 --hs_mask_proportion=0.75  --num_classes=19 --from_scratch=True --target_modalities=bimodal > mae_trainer_finetuning.log &  
```

Following models are available:
- mae_vit_tiny_patch24
- mae_vit_small_patch24
- mae_vit_base_patch24

## Citation
If you find this code useful in your research, please consider citing:
```
@inproceedings{kukushkin2024bimae,
  author={Kukushkin, Maksim and Bogdan, Martin and Schmid, Thomas},
  booktitle={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)}, 
  title={BiMAE - A Bimodal Masked Autoencoder Architecture for Single-Label Hyperspectral Image Classification}, 
  year={2024},
  pages={2987-2996},
  keywords={Manifolds;Visualization;Costs;Scalability;Conferences;Self-supervised learning;Pattern recognition;masked autoencoder;hyperspectral imaging;seed purity testing;hyperspectral classification;multimodal masked autoencoder;masked modeling;self-supervised learning},
  doi={10.1109/CVPRW63382.2024.00304}}
```
