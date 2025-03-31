
## AI Detection + Auto-Captioning for Transparency in Image Platforms

<!-- 
Discuss: Value proposition: Your will propose a machine learning system that can be 
used in an existing business or service. (You should not propose a system in which 
a new business or service would be developed around the machine learning system.) 
Describe the value proposition for the machine learning system. What’s the (non-ML) 
status quo used in the business or service? What business metric are you going to be 
judged on? (Note that the “service” does not have to be for general users; you can 
propose a system for a science problem, for example.)
-->
Proposing a system that can be used in existing image-sharing platforms (Pinterest, Instagram) to increase transparency by using an ML model to detect AI-generated images and enabling auto-captioning to tag images appropriately

**Value Proposition:** The platforms will be able to moderate content better/prevent misinformation using the tags provided by the model. Additionally the automatic tagging mechanism can also be used in content indexing.

**Status Quo:** Users are consuming content with no way of knowing if the content is AI-generated or not. Content is being uploaded with manual captions that may/may not help with information retrieval/censorship.

**Business Metric:** Preventing misinformation by informing users. Faster content retrieval.

### Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->

| Name                            | Responsible for | Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| All team members | | |
| Ansh Sarkar | Model Training (Unit 4&5)  |  |
| Manali Tanna | Model Serving & Monitoring (Unit 6&7) | |
| Princy Doshi | Data Pipeline (Unit 8) | |
| Simran Kucheria |  Continuous Pipeline (Unit 3) ||



### System diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

|              | How it was created | Conditions of use |
|--------------|--------------------|-------------------|
| AI vs. Human-Generated Images|  Images sampled from the Shutterstock platform across various categories, including a balanced selection where one-third of the images feature humans. These authentic images are paired with their equivalents generated using state-of-the-art generative models. |Licensed under Apache 2.0 - a permissive open-source license that allows users to modify, distribute, and sublicense the original code, but requires including the original copyright notice, a copy of the license, and any significant changes made to the code.                   |
| MS COCO Dataset  |   The MS COCO (Microsoft Common Objects in Context) dataset is a large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images. |  Licensed under Creative Commons Attribution 4.0 License, lets you distribute, remix, tweak, and build upon your work, even commercially, as long as you credit the original creator.|
|Resnet50 |ResNet50 is a convolutional neural network (CNN) with 50 layers, part of the ResNet family introduced by Microsoft Research in 2015. It uses "residual learning" via skip connections to solve vanishing gradient problems in deep networks. It is pre-trained on the ImageNet-1k dataset (1.2M images, 1,000 classes).|It is  open-source under MIT License. It is free for research and commercial use. Requires attribution if used directly.The pretrained weights may have dataset-specific biases.|
|ViT (Vision Transformer)| Introduced by Google in 2020, ViT applies the transformer architecture (originally for NLP) to images by splitting images into patches.It is pretrained on large datasets like ImageNet-21k (14M images) or JFT-300M (proprietary dataset).|The Original ViT code/weights is under Apache 2.0 License (commercial-friendly). Third-party implementations (e.g., Hugging Face) may have specific terms. Pretrained models may inherit biases from training data. Proprietary datasets (e.g., JFT-300M) are not publicly available.|
|Qwen/Qwen2-VL-7B-Instruct| A 7-billion-parameter multimodal vision-language model from Alibaba’s Qwen series. Combines vision and language transformers for tasks like visual QA and instruction following.|Governed by Tongyi Qianwen License (check Hugging Face for specifics). Non-commercial/research use only unless explicitly permitted. Prohibited for military, surveillance, or unethical applications. Users must comply with local laws (e.g., China’s AI regulations).|


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100`     | 4 hour block twice a week                         |               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

#### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->


