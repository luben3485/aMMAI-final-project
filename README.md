# NTU-AMMAI-CDFSL

## Experimental Results
   https://docs.google.com/document/d/1Y_RRiJRqP8v4OpCuIm4wL7_09j4jQrWpluvWuVGEbtI/edit?usp=sharing

## First Steps
   Download all trained baseline models and datasets via this link:
   https://drive.google.com/drive/folders/14RD6uP2iik8fQoFz2W1baKJ2eK2rih-y?usp=sharing
   
## Tracks
   **track 1. Few-shot learning (fsl)**
   
   - source domain: mini-ImageNet
   - target domain: mini-ImageNet
   - 5-way 5-shot in meta-test.

   **track 2. Cross-domain Few-shot learning with single source domain (cdfsl-single)**
   
   - source domain: mini-ImageNet
   - target domain: CropDisease, EuroSAT, ISIC
   - 5-way 5-shot in meta-test.

   **track 3. Cross-domain Few-shot learning with multiple source domains (cdfsl-multi)**
   
   - source domain: mini-ImageNet, cifar-100
   - target domain: CropDisease, EuroSAT, ISIC
   - 5-way 5-shot in meta-test.

## Requirements and Steps
   1. Download all the needed datasets and trained model via above link and place them to correct locations.
   2. Change configuration in config.py to the correct paths in your own computer if needed.

# Training and Testing Commands
   - **Command for training the model**
      ```bash
           python train.py --task fsl/cdfsl-single --model MODEL  --method METHOD --n_shot 5 --train_aug
           python train.py --task cdfsl-multi --model MODEL  --method METHOD --n_shot 5 --train_aug
       ```
       - METHOD: protonet(baseline), e_protonet_fc, and e_relationnet_fc.
       - TASK: [fsl/cdfsl-single/cdfsl-multi], please note that selecting fsl and cdfsl-single is exactly same in trainig (only trained on mini-ImageNet, so just choose one of them, not both).
       - n_shot: how many samples per class in training tasks, **you can modify this but the maximum should be less than 10.**
       - MODEL: ResNet10 for now, there are many other models in backbone.py, you can change them by yourself.
       - train_aug: it's optional, apply augmentation to training data (suggested).
       
   - **Command for testing the model**
        ```bash
            python meta_test_few_shot_models.py --task TASK --model MODEL --method METHOD  --train_aug --finetune
        ```
       - METHOD: protonet(baseline), e_protonet_fc, and e_relationnet_fc.
       - TASK: [fsl/cdfsl-single/cdfsl-multi], three types of tasks, you need to select each of them.
       - MODEL: ResNet10 for now, there are many other models in backbone.py, you can change them by yourself.
       
       - train_aug: you need to add this if you use it in the training (for finding the correct path).
       - finetune: (optional) use the PQS to finetune the model if selected.
