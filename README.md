
## DeepTrust: AI-Generated Media Detection & Smart Tagging

<!-- 
Discuss: Value proposition: Your will propose a machine learning system that can be 
used in an existing business or service. (You should not propose a system in which 
a new business or service would be developed around the machine learning system.) 
Describe the value proposition for the machine learning system. What’s the (non-ML) 
status quo used in the business or service? What business metric are you going to be 
judged on? (Note that the “service” does not have to be for general users; you can 
propose a system for a science problem, for example.)
Main focus is does it make us some money, should project be approved
Scale Requirement : List dataset size, list model size, no of parameters
Value Proposition : I want you to tell me exactly who is the customer who I'm gonna sell this to. Who is the ONE customer, name them. It can be a made up name like EXinterest not necessirly "Pinterest". Make sure they are suitable for the suitable context they're deployed in. To what extent can we satify the requirements of the customer. Tell something about the customer. What number of posts do they get in a day. NAME THE CUSTOMER. 
She also needs specific data about their volume and how we satisfy it
Status Quo: Name the business, give specifics, whats the current situation
Outside Materials: provide link, enumerate every dataset, link to a paper who created, when where how of the dataset
Mention dataset sizes, Answer her questions also, is it an ongoing?
Privacy/Fairness/Ethics Issues -  think about it. Either data or the system itself. The system itself has potential to cause harm, for example: chest Xrays
Business Metrics: Throughput, latency etc dont count. Click Through Rate, More customers will buy, more customers will spend less time doing something
-->

Proposing a system that can be used inside Flickr to increase transparency by using an ML model to detect/flag AI-generated images for potential review and enabling LLM based auto-captioning to tag images for indexing.

**Value Proposition:** By incorporating AI vs. human image detection, Flickr can provide users with authenticity in visual content, helping them distinguish between real and AI-generated images. This is particularly valuable for photographers who rely on Flickr for genuine community shared content. Additionally, auto-captioning and tagging allowing the Flickr team to tag images more efficiently and thereby enhancing searchability. Moreover, the description and tags generated will be meaningful opening up avenues for future work. These improvements can boost user engagement, content organization, and accessibility, making Flickr a popular platform.

<!-- The platform will be able to moderate content better and prevent misinformation using the tags provided by the model. Additionally, the automatic tagging mechanism can also be used in content indexing. -->

**Status Quo:** Users are consuming content on Flickr without any indication of whether an image is AI-generated or authentic, which can lead to misinformation and difficulty in verifying sources. We have noticed articles online [(Reddit: AI Content is Putting Me Off)](https://www.reddit.com/r/flickr/comments/1bajy3p/ai_content_is_putting_me_off/) of users expressing frustration over increasing number of AI generated images on Flickr. Currently, content is uploaded with manual captions that may or may not be accurate, detailed, or useful for search and discovery. Many users either leave captions blank, use vague descriptions, or provide misleading tags, making it difficult to retrieve relevant images efficiently. Additionally, without standardized metadata, content moderation and manually identifying inappropriate or AI-generated content at such scale is impractical.

Flickr boasts of 10billion photos shared since inception with about 25 million photos shared daily. That translates to a concurrency requirement of 300/second. The system we are proposing right now can handle upto 50 requests/second with the capability of scaling more.

<!-- 
Users are consuming content with no way of knowing if it is AI-generated or not. Content is being uploaded with manual captions that may or may not help with information retrieval or censorship. -->

**Business Metric:** 
- Preventing Misinformation
  - By clearly labeling AI-generated images, Flickr can help users make informed decisions about the content they consume, increasing trust and transparency of Flickr.
- Faster Content Retrieval
  - AI-powered auto-captioning and tagging improves searchability, enabling users to find relevant images more quickly, thereby enhancing user experience and engagement on Flickr.

### Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->

Team Name: 

| Name            | Responsible for                       | Link to their commits in this repo |
| --------------- | ------------------------------------- | ---------------------------------- |
| Team AMPS!      |                                       |                                    |
| Ansh Sarkar     | Model Training (Unit 4&5)             |                                    |
| Manali Tanna    | Model Serving & Monitoring (Unit 6&7) |                                    |
| Princy Doshi    | Data Pipeline (Unit 8)                |                                    |
| Simran Kucheria | Continuous Pipeline (Unit 3)          |                                    |



### System diagram

<!-- Overall diagram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->

<!-- Insert image here -->
![System Diagram](images/System%20Diagram.png)

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

|                               | How it was created | Conditions of use  |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AI vs. Human-Generated Images | Images sampled from the Shutterstock platform across various categories, including a balanced selection where one-third of the images feature humans. These authentic images are paired with their equivalents generated using state-of-the-art generative models. Link to dataset: https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset Size of Dataset: 13Gb| Licensed under Apache 2.0 - a permissive open-source license that allows users to modify, distribute, and sublicense the original code, but requires including the original copyright notice, a copy of the license, and any significant changes made to the code.|
| MS COCO Dataset               | The MS COCO (Microsoft Common Objects in Context) dataset is a large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images. Link to dataset: https://cocodataset.org/#home Size of Dataset: 20Gb| Licensed under Creative Commons Attribution 4.0 License, lets you distribute, remix, tweak, and build upon your work, even commercially, as long as you credit the original creator.   |
| Flickr Image Dataset                        | A new benchmark collection for sentence-based image description and search, consisting of 30,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. … The images were chosen from six different Flickr groups, and tend not to contain any well-known people or locations, but were manually selected to depict a variety of scenes and situations Link to dataset: https://www.kaggle.com/datasets/adityajn105/flickr30k Size of Dataset: 5Gb |Licensed under Creative Commons Attribution 4.0 License, lets you distribute, remix, tweak, and build upon your work, even commercially, as long as you credit the original creator. |                                                                                   
| RegNet                        | RegNet (Regularized Network) is a family of CNNs by Facebook AI, designed for efficient and scalable deep learning. It optimizes network design systematically, using adjustable parameters like depth and width. RegNet balances accuracy and computational cost, making it ideal for tasks like image classification and object detection. Link to Paper: https://arxiv.org/pdf/2101.00590  | The RegNet model trained on ImageNet-1K, introduced in the paper "Designing Network Design Spaces," is available on Hugging Face under the Apache 2.0 License. This permissive license allows for both personal and commercial use, distribution, and modification, provided that proper attribution is given and a copy of the license is included with any distributions. |                                                                             
| Qwen/Qwen2-VL-7B              | A 7-billion-parameter multimodal vision-language model from Alibaba’s Qwen series. Combines vision and language transformers for tasks like visual QA and instruction following. Link to reference: https://huggingface.co/Qwen/Qwen2-VL-7B | Governed by Tongyi Qianwen License (check Hugging Face for specifics). Non-commercial/research use only unless explicitly permitted. Prohibited for military, surveillance, or unethical applications. Users must comply with local laws (e.g., China’s AI regulations).| 


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, and persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement                             | How many/when                                     | Justification |
| --------------------------------------- | ------------------------------------------------- | ------------- |
| `m1.medium` VMs                         | 3 for entire project duration                     | 1 for training/retraining purposes, 2 for serving inference(1 will host models(GPU) and the other will be used to expose the application)              |
| `⁠gigaio-compute-07` or  `P3-GPU-009 ⁠`   | 4 hour block thrice a week                        | Training/Retraining LLM/RegNet            |
| Floating IPs                            | 1 for entire project duration, 2 for sporadic use | Expose Production env, other 2 for canary/staging environments              |
| Fast SSD(Persistent Storage)            | 512Gb Duration of entire project                   |Datasets are around ~40Gb (27+12) + Models are huge (8.29B parameters for LLM + 145M parameters for RegNet) ~ 18Gb |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material. 
(5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->
##### Objectives
- **Train and re-train**: The system will use the AI vs Human Images dataset and train a RegNet model to classify whether the image is AI generated or not. Parallely we will use an LLM and finetune it on the COCO dataset such that it generates description for the image, generate tags for it which can be used further for indexing. For the re-train pipeline both the models can have periodic retraining jobs scheduled after a certain amount of data has been annotated. 
  
- **Modeling**: We have two tasks to solve. First one being an image classification problem we are going ahead with either a RegNet model or a ViT model as these models are known to perform well in image classification tasks and we can use their pretrained weights to finetune for our use case. For image captioning as the second task we have decided to go ahead with a multimodel LLM model (Qwen/Qwen2-VL-7B) as it is pretrained on a large dataset and can be finetuned for image captioning. The pretrained knowledge of the model will help us in generating more meaningful captions and tags.
  
- **Experiment tracking**: Both the model will be tracked using MLFlow to track model experiments, hyperparameters and metrics. This will help us in logging the model performance and also help us in comparing the models. We will also use MLFlow to track the training jobs and their artifacts. Some of the experiments that we are planning to run are:
    - For RegNet Classification Model:
      - Different architectures changes such as adding dropouts, playing around with input size etc.
      - Different optimizers such as Adam, SGD, AdamW etc.
      - Different learning rates and schedulers.
      - Different augmentation techniques such as random cropping, flipping, rotation etc.
      - Different training strategies such as gradient accumulation and mixed precision.
    - For LLM Captioning Model:
      - Different multi-modal architectures such as Qwen/Qwen2-VL-7B, Llama-3.2-11B-Vision-Instruct
      - Different parameters for LoRA and QLoRA for finetuning.
  
- **Scheduling training jobs**: All the jobs required for training/re-training will be submitted via a ray cluster.

##### Extra Difficulty

- **Training strategies for large models**: The LLM/Resnet models are too large to fit on a low end GPU. Hence some strategies like PEFT, gradient accumulation and mixed precision will be used whilst training. These experiments will be tracked using MLFLow to come up with the most optimum model.
  
- **Use distributed training to increase velocity**: We will experiment with FSDP/DDP techniques whilst running experiments on our model.
  
- **Using Ray Train**: We will use Ray-Train to ensure frequent checkpointing and guarantee fault tolerance whilst training.

- **Scheduling hyperparameter tuning jobs**: We will use Ray Tune to schedule hyperparameter tuning jobs for the RegNet model for classification task. 


#### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->
##### Objectives

- **Serving from an API endpoint**: The system will be exposed to users through REST API endpoints. We will use a basic SwaggerUI-based backend. The image tags and description will be sent back as a response to different internal stakeholders who can then use the tags for content moderation and indexing. 
  
- **Identify requirements**: 
  - The system must be highly available and scalable to support global user traffic, including unexpected spikes in usage. However the system does not have to provide inference in real time and can afford to have a delay since the system is used internally and does not directly face the user.
  - Flickr handles approximately 25 million uploads per day, translating to an average of ~300 image uploads per second. The proposed system currently supports up to 50 requests per second, with built-in auto-scaling capabilities to accommodate peak loads.
  - The ideal model inference time (for AI detection and captioning) should be ≤ 5 minutes per request. The API gateway should introduce no more than 200ms of latency. The total end-to-end response time, from image upload to processed output, should not exceed 10 minutes to ensure a scalable and accurate system.
  
- **Model optimizations to satisfy requirements**: Since our system has real-time low-latency requirements and expects a large volume of concurrent users, we will experiment with different model-level optimizations like quantization, and reduced precision using the ONNX backend to keep our inference time as low as possible for our RegNet model. For the LLM we will use either Huggingface Accelerate or vLLM that takes advantage of PagedAttention, continuous batching, and tensor parallelism to reduce the latency of the model. We can also quantize the LLM to reduce the size of the model and improve the inference time.
  
- **System optimizations to satisfy requirements**: To further reduce our prediction latency, we will experiment with various system-level optimizations for model serving, including concurrency and batching. We will use different execution providers like Triton, TensorRT, ONNX for our RegNet model.

- **Offline evaluation of model**: Our system will run an automated offline evaluation plan after model training, with results logged to MLFlow. The offline evaluation will include: 

  - Evaluation on appropriate domain specific metrics for each model. For AI VS Human Image detection using a RegNet we will use F1 score, Precision, Accuracy and Confusion Matrix. For Image description and tagging using an LLM we will use BLEU scores.
  - We will evaluate the model on different populations of images such as images with humans, animals, objects etc. We will also evaluate the model on different slices of data such as images with different resolutions, images with different colors, images with different backgrounds etc.
  - Test on known failure modes where we will try the model on images that are known to be difficult such as low resolution/noisy images, artistic renditions of real images that are not AI generated etc. We will create a specialized test set containing hand-drawn illustrations, paintings and artistic photographs with filters, collages and mixed-media art, AI-generated images mimicking artistic styles like Ghibli Images. We will also test for edge case scenarios for example: Hybrid images that contain both AI and human-created elements, Heavily photoshopped images, Images featuring uncommon or surreal elements etc.
  - Unit tests based on templates. We will create a set of unit tests comprising of images that we have an expected output for. These tests will be run on the model to check if the model is working as expected. We will also create a set of unit tests for the API endpoints to check if the API is working as expected.

- **Load test in staging**: We will do a load test for each of the model separately and also for the entire system. This will help us identify if each component is working as expected and individual performance analysis can help us find any bottlenecks in our system.
   
- **Online evaluation in Canary**: Simulate data patterns corresponding to different category of users - multiple posts, misinformation, model users etc. The data range of these users to represent all the different categories of data seen so far.

- **Close the loop**: We plan to close the loop in various ways.
    - For the AI vs Human Image detection model, we will use the feedback from the users. All the images that were tagged as AI generated will be sent to a human moderator who will verify if the image is AI generated or not and a random sample of images that were not tagged as AI generated will also be sent to a human moderator for verification. This feedback loop will help us add more data for both the classes.
    - For image captioning and tagging, we will use human annotators to annotate a random sample of images and add it to our training dataset. This will keep our model up to date with any latest trends and changes in the data.

- **Business-specific evaluation**: The business specific evaluation for this system can be done in the following ways
    - A metric to evaluate the percentage of correctly tagged images with respect to AI-generated content.
    - A metric like Click Through Rate (CTR) to evaluate how quickly is useful content being returned with the use of information retrieval mechanisms.

##### Extra Difficulty
- **Develop multiple options for serving**: The RegNet model benefits from using GPU for inference and we plan to develop and evaluate an optimized server-grade GPU. OpenVINO execution provider should also be beneficial for our system. We will experiment with all the execution providers and compare the performance of our system.
  
- **Monitor for data drift**: Given that our model will work with images, we will also monitor for data drift for images using feature embeddings and computing KL divergence on it.
  
- **Monitor for model degradation**: We will monitor for model degradation in the model output by closing the feedback loop. We will trigger automatic model re-training with the new image and its provided label.

#### Data pipeline
<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->
##### Objectives
- Persistent storage: model training artifacts, test artifacts, models, container images, and data artifacts to be stored on provisioned persistent data storage.

- Offline data: 80% of both the datasets outlined above will be used as offline data stored in the above provisioned data storage.

- Data pipelines: Our data pipeline will ingest image data from multiple data sources including our AI vs. Human dataset and MS COCO. The pipeline will also clean the data such that it is ready to use for model training. The original versions will be stored in the raw data storage. We'll also employ dataset versioning. We will perform data quality checks to validate incoming images for dimensions, and format compliance before entering the pipeline. For production feedback, we'll implement a closed-loop system where user reports and corrections are automatically fed back into our training datasets after human verification.

- Online data: A script to simulate data consisting of images, some AI-generated, some real.


#### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->


##### Objectives
- Infrastructure-as-code: The hardware requirements for our project will be provisioned using YAML and terraform. The software/service setup will be configured using ArgoCD and Argo workflows. All these configurations will live on git. 

- Cloud-native: We will ensure immutability in our infrastructure by using IaC and as seen in our design diagram the RegNet and LLM's will be two separate microservices having separate API endpoints. All code will be containerized and run on docker/K8.

- CI/CD and continuous training: 
A CI/CD pipeline will be defined on Gitlab with configurations for all of the other pipelines. We will also incorporate our IAC configs here. The conditions for retraining will include - Changes in the data versioning, changes in model code and a weekly training schedule. This continuos X pipeline will go through all the defined pipelines - training, serving, evaluation, data automatically.

<!-- You will define an automated pipeline that, in response to a trigger (which may be a manual trigger, a schedule, or an external condition, or some combination of these), will: re-train your model, run the complete offline evaluation suite, apply the post-training optimizations for serving, test its integration with the overall service, package it inside a container for the deployment environment, and deploy it to a staging area for further testing (e.g. load testing). -->

- Staged deployment: We will configure “staging”, “canary”, and “production” environments on which our service may be deployed. We will use a combination of kubernetes, MLFlow, Argo Workflows and github actions for this. You will also implement a process by which a service is promoted from the staging area to a canary environment, for online evaluation, and a process by which a service is promoted from canary to production.
