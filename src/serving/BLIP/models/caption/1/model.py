import os
import io
import base64
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import triton_python_backend_utils as pb_utils
from utils import blip_decoder

class TritonPythonModel:
    
    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        model_path = os.path.join(model_dir, "model_base_caption_capfilt_large.pth")
        
        instance_kind = args.get("model_instance_kind", "cpu").lower()
        if instance_kind == "gpu":
            device_id = int(args.get("model_instance_device_id", 0))
            torch.cuda.set_device(device_id)
            self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        # self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = blip_decoder(pretrained=model_path, image_size=384, vit='base')
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
        transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])


    def preprocess(self, image_data):
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)

        if isinstance(image_data, bytes):
            image_data = image_data.decode("utf-8")
            image_data = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_data)).convert('RGB')


        img_tensor = self.transform(image).unsqueeze(0)
        return img_tensor

    def execute(self, requests):
        # Gather inputs from all requests
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            input_data_array = in_tensor.as_numpy()  # each assumed to be shape [1]
            # Preprocess each input (resulting in a tensor of shape [1, C, H, W])
            image = self.preprocess(input_data_array[0,0]).to(self.device)
            # batched_inputs.append(self.preprocess(input_data_array[0, 0]))
            with torch.no_grad():
                caption = self.model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)

            # Convert the caption to a numpy array
            caption_np = np.array(caption, dtype=object)
            # Create a Triton tensor for the caption
            out_tensor = pb_utils.Tensor("CAPTION", caption_np)
            # Create the inference response
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)
        return responses
        
        # # Combine inputs along the batch dimension
        # batched_tensor = torch.cat(batched_inputs, dim=0).to(self.device)
        # print("BatchSize: ", len(batched_inputs))
        # # Run inference once on the full batch
        # with torch.no_grad():
        #     outputs = self.model(batched_tensor)
        
        # # Process the outputs and split them for each request
        # responses = []
        # for i, request in enumerate(requests):
        #     output = outputs[i:i+1]  # select the i-th output
        #     prob, predicted_class = torch.max(output, 1)
        #     predicted_label = self.classes[predicted_class.item()]
        #     probability = torch.sigmoid(prob).item()
            
        #     # Create numpy arrays with shape [1, 1] for consistency.
        #     out_label_np = np.array([[predicted_label]], dtype=object)
        #     out_prob_np = np.array([[probability]], dtype=np.float32)
            
        #     out_tensor_label = pb_utils.Tensor("FOOD_LABEL", out_label_np)
        #     out_tensor_prob = pb_utils.Tensor("PROBABILITY", out_prob_np)
            
        #     inference_response = pb_utils.InferenceResponse(
        #         output_tensors=[out_tensor_label, out_tensor_prob])
        #     responses.append(inference_response)
        
        # return responses