# NTU-AMMAI-CDFSL

The source code of NTU-AMMAI-CDFSL 2021 version.

## First Steps
   Download all trained baseline models and datasets via this link:
   https://drive.google.com/drive/folders/14RD6uP2iik8fQoFz2W1baKJ2eK2rih-y?usp=sharing

## Description
   https://docs.google.com/document/d/1VtVR45wBvQlSnap1AArqIvnjoUQeKdWFkWe9EksLEIw/edit?usp=sharing

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
   3. **(requirement)** Meta-test the trained Baseline and ProtoNet on three tasks and report the results, more precisely, run the following commands.
      ```bash
         python meta_test_Baseline.py --task fsl --model ResNet10 --method baseline  --train_aug
         python meta_test_Baseline.py --task cdfsl-single --model ResNet10 --method baseline  --train_aug 
         python meta_test_Baseline.py --task cdfsl-multi --model ResNet10 --method baseline  --train_aug
         python meta_test_few_shot_models.py --task fsl --model ResNet10 --method protonet  --train_aug
         python meta_test_few_shot_models.py --task cdfsl-single --model ResNet10 --method protonet  --train_aug --finetune
         python meta_test_few_shot_models.py --task cdfsl-multi --model ResNet10 --method protonet  --train_aug --finetune
      ```
      You need to run these commands now, because it takes **A LOT OF TIME**.
   4. **(requirement)** Design a metric-learning based model (inherit the template in meta_template.py) and solve the three tasks. 
   5. **(requirement)** Design a training method on dealing with multiple source domains. You can directly use your method to train your model in step 4 (in cdfsl-multi task).
   6. (optional) You are encouraged to apply any speed-up improvements during both training and testing, just make sure you don't modify the evaluation protocol.
   7. (optional) We have provided a fine-tuning method for metric-learning based model, you can simply use it or design your own ones. But please make sure you can't access query set in fine-tuning.
# Training and Testing Commands - Details 
   - **Command for training the baseline model**

       ```bash
           python train.py --task TASK --model MODEL  --method baseline --train_aug
       ```
       
       - TASK: [fsl/cdfsl-single/cdfsl-multi], please note that selecting fsl and cdfsl-single is exactly same in trainig (only trained on mini-ImageNet, so just choose one of them, not both).
       - MODEL: ResNet10 for now, there are many other models in backbone.py, you can change them by yourself.
       - train_aug: it's optional, apply augmentation to training data (suggested).
       
   - **Command for training the metric-learning based model**

       ```bash
           python train.py --task TASK --model MODEL  --method METHOD --n_shot 5 --train_aug
       ```
       
       - TASK: [fsl/cdfsl-single/cdfsl-multi], please note that selecting fsl and cdfsl-single is exactly same in trainig (only trained on mini-ImageNet, so just choose one of them, not both).
       
       - MODEL: ResNet10 for now, there are many other models in backbone.py, you can change them by yourself.
       - METHOD: protonet, and **YOUR METHOD**.
       - n_shot: how many samples per class in training tasks, **you can modify this but the maximum should be less than 10.**
       - train_aug: it's optional, apply augmentation to training data (suggested).

     - **TODO, you need to train your model by the following commands:**
       
       ```bash
           python train.py --task fsl/cdfsl-single --model MODEL  --method YOUR_METHOD --n_shot 5 --train_aug
           python train.py --task cdfsl-multi --model MODEL  --method YOUR_METHOD --n_shot 5 --train_aug
       ```
   - **Command for testing the baseline model**
        
       - For Baseline, we will train a new linear classifier using support set (a kind of finetuning) before inference when solveing all three tasks.

        ```bash
            python meta_test_Baseline.py --task TASK --model MODEL --method baseline  --train_aug --freeze_backbone
        ```
       - TASK: [fsl/cdfsl-single/cdfsl-multi], three types of tasks, you need to select each of them.
       - MODEL: ResNet10 for now, there are many other models in backbone.py, you can change them by yourself.
       - METHOD: baseline.
       - train_aug: you need to add this if you use it in the training (for finding the correct path).
       - freeze_backbone: (optional) the backbone(ResNet10 in default) will be frozen during fine-tuning if you select this. 

   - **Command for testing the metric-learning based model**
      
       - For metric-learning based models, we apply a pseudo query set (PQS) to fine-tune the model before inference when solveing cdfsl-related (task 2 and 3) tasks.

        ```bash
            python meta_test_few_shot_models.py --task TASK --model MODEL --method METHOD  --train_aug --finetune
        ```
       - TASK: [fsl/cdfsl-single/cdfsl-multi], three types of tasks, you need to select each of them.
       - MODEL: ResNet10 for now, there are many other models in backbone.py, you can change them by yourself.
       - METHOD: protonet and **YOUR METHOD**.
       - train_aug: you need to add this if you use it in the training (for finding the correct path).
       - finetune: (optional) use the PQS to finetune the model if selected, you can design your own fine-tuning method, just make sure you **CAN'T** access query set in testing.

   When you meta-testing a model, a dataset contains 600 tasks.

   After evaluating 600 times, you will see the result like this: 600 Test Acc = 49.91% +- 0.44%.

## Result Table

You need to provide a table like this in your report.

<table>
    <thead>
        <tr>
            <th rowspan=2>Models</th>
            <th>fsl</th>
            <th colspan=3>cdfsl-single</th>
            <th colspan=3>cdfsl-multi</th>
        </tr>
        <tr>
            <th>mini-ImageNet</th>
            <th>CropDisease</th>
            <th>EuroSAT</th>
            <th>ISIC</th>
            <th>CropDisease</th>
            <th>EuroSAT</th>
            <th>ISIC</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Baseline</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
        </tr>
        <tr>
            <td>ProtoNet</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
        </tr>
        <tr>
            <td>YOUR METHOD</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
            <td>XX% ± yy%</td>
        </tr>
    </tbody>
</table>

You can provide more results if you have multiple variants.

Also, you can put result with () in baseline methods cells if you have tried different settings. E.g. add --freeze_backbone or not wehn testing Baseline.

At the end, you can also provide the result of baseline models with your training method in cdfsl-multi task, if your training method is suitable for the baseline methods.

## Contact Information
TA: Jia-Fong Yeh (jiafongyeh@ieee.org)

## Reference
This repository is modified from the following repos.
- https://github.com/wyharveychen/CloserLookFewShot
- https://github.com/IBM/cdfsl-benchmark
