
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
|Qwen/Qwen2-VL-7B| A 7-billion-parameter multimodal vision-language model from Alibaba’s Qwen series. Combines vision and language transformers for tasks like visual QA and instruction following.|Governed by Tongyi Qianwen License (check Hugging Face for specifics). Non-commercial/research use only unless explicitly permitted. Prohibited for military, surveillance, or unethical applications. Users must comply with local laws (e.g., China’s AI regulations).|


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

<!-- Insert image here -->
![System Diagram](images/System%20Diagram.png)


#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->
##### Objectives
- Train and re-train: The system will use the AI vs Human Images dataset and train a ResNet model to classify whether the image is AI generated or not. Parallely we will use an LLM and finetune it on the COCO dataset such that it generates description for the image, generate tags for it and also provide a decision on whether or not it should be content moderated. The Resnet model will be retrained after the feedback loop closes and new data (image + label) is ingested.
- Modeling: The Resnet model will involve modelling choices with respect to hyperparamters. For image description generation we choose to go ahead with an LLM because of the way these models generalize well on text generation and we can prompt engineer it to perform well for content moderation. (TODO: @ansh Add reason why ResNet)
- Experiment tracking: We plan to run multiple training jobs for the ResNet model with different hyperparameters. All these jobs will be tracked on MLFlow. The LLM model will also have fine-tuning jobs associated with it. Some of the experiments we plan to run are: 

    1. LLM

        1. Finetuning on COCO dataset
        2. Prompt Engineering
        3. Hyperparameter tuning

    2. ResNet

        1. Architecture
        2. Learning Rate
        3. Dropout
        4. Cosine annealing

- Scheduling training jobs: All the jobs required for training/re-training will be submitted via a ray cluster.

##### Extra Difficulty
- Training strategies for large models: The LLM/Resnet models are too large to fit on a low end GPU. Hence some strategies like PEFT, gradient accumulation and mixed precision will be used whilst training. These experiments will be tracked using MLFLow to come up with the most optimum model.
- Use distributed training to increase velocity: We will experiment with FSDP/DDP techniques whilst running experiments on our model.
- Using Ray Train: We will use Ray-Train to ensure frequent checkpointing and guarantee fault tolerance whilst training.


#### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->
##### Objectives

- Serving from an API endpoint: The system will be exposed to users through REST API endpoints. We will use a basic SwaggerUI based backend. Since this system is intended for internal use we are not proposing a frontend.
- Identify requirements: [Based on platform come up with concurrency/latency requirements, also talk about model size if we plan to have edge devices host the model?] (TODO @all)
- Model optimizations to satisfy requirements: Since our system has real time low latency requirements and expects a large volume of concurrent users, we will experiment with different model-level optimisations like graph optimization, quantization and reduced precision using ONNX backend to keep our inference time as low as possible.
- System optimizations to satisfy requirements: To further reduce our prediction latency we will experiment with various system level optimizations for model serving, including concurrency and batching, in Triton Inference Server.

- Offline evaluation of model: Our system will run an automated offline evaluation plan after model training with results logged to MLFlow. The offline evaluation will include: 
(A) evaluation on appropriate domain specific metrics for each model. For AI VS Human Image detection using a ResNet we will use F1 score, Precision, Accuracy and Confusion Matrix. For Image description, content moderation and tagging using an LLM we will use BLEU scores.
(B) evaluation on populations and slices of special relevance, including an analysis of fairness and bias if relevant (TODO @all if we do content moderation we have to add something here) 
(C) test on known failure modes 
(D) and, unit tests based on templates. Depending on the test results, you will automatically register an updated model in the model registry, or not.
- Load test in staging: 
- Online evaluation in canary: 
- Close the loop: 
- Business-specific evaluation: 

##### Extra Difficulty
- Develop multiple options for serving: The ResNet model benefits from using GPU for inference and we plan to develop and evaluate optimized server-grade GPU. We will compare them with respect to performance and cost.
- Monitor for data drift: Given that our model will work with images, we will also monitor for data drift for unexpected and undocumented changes to the data structure and semantics.
- Monitor for model degradation: We will monitor for model degradation in the model output by closing the feedback loop. We will trigger automatic model re-training with the new image and it's provided label.

#### Data pipeline
<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->
##### Objectives
- Persistent storage: model training artifacts, test artifacts, models, container images, data artifacts to be stored on provisioned persistent data storage.

- Offline data: 80% of both the datasets outlined above will be used as offline data stored in the above provisioned data storage.

- Data pipelines: idk this guys

- Online data: A script to simulate data consisting of images, some AI generated, some real.
#### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->

#TO BE UPDATED AFTER LAB 3 IS OUT

##### Objectives
- Infrastructure-as-code: You will avoid ClickOps in provisioning your infrastructure. Instead, you’ll define your infrastructure configuration in version control using either a declarative style (like Terraform, which we use in Lab 3) or imperative style (like python-chi, used in many other labs). This configuration must live in Git. You will similarly avoid manual installation and configuration of your infrastructure, once it is deployed. Instead, you can use a combination of automation tools like Ansible (used in Lab 3), ArgoCD (used in Lab 3), Argo Workflows (used in Lab 3), Helm, python-chi, and/or other tools to set up your infrastructure. Like your infrastructure configuration, these service and software configurations similarly must live in Git.

- Cloud-native: You will be expected to develop your project as a “cloud-native” service. This means: (1) Immutable infrastructure: you avoid manual changes to infrastructure that has already been deployed. Instead, make changes to your configurations in Git, and then bring up your new infrastructure directly from these configurations. (2) Microservices: to the extent that is reasonable, deploy small independent pieces that work together via APIs, but are managed separately. (3) Containers as the smallest compute unit: you will containerize all services, so they can be deployed and scaled efficiently. You won’t run anything (except building and managing containers!) directly on compute instances.

- CI/CD and continuous training: You will define an automated pipeline that, in response to a trigger (which may be a manual trigger, a schedule, or an external condition, or some combination of these), will: re-train your model, run the complete offline evaluation suite, apply the post-training optimizations for serving, test its integration with the overall service, package it inside a container for the deployment environment, and deploy it to a staging area for further testing (e.g. load testing).

-Staged deployment: You will configure a “staging”, “canary”, and “production” environment in which your service may be deployed. You will also implement a process by which a service is promoted from the staging area to a canary environment, for online evaluation; and a process by which a service is promoted from canary to production.
