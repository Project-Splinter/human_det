import tqdm
import cv2
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from .utils import dboxes300_coco, Encoder

# Detection
class Detection:
    def __init__(self, device="cuda:0", fp16=False):
        self.device = device
        self.fp16 = fp16
        self.utils = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
        self.model = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', 
            model_math='fp16' if self.fp16 else 'fp32'
        ).to(device)
        self.model.eval()

        self.classes_to_labels = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
            'bus', 'train', 'truck', 'boat', 'traffic light', 
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        dboxes = dboxes300_coco(device)
        self.encoder = Encoder(dboxes)

    def prepare_input(self, tensor):
        _, _, H, W = tensor.size()
        if H > W:
            dstH, dstW = 300, int(300/H * W)
            padH, padW = [0, 0], [(300-dstW)//2, 300-dstW-(300-dstW)//2]
        elif H < W:
            dstH, dstW = int(300/W * H), 300
            padH, padW = [(300-dstH)//2, 300-dstH-(300-dstH)//2], [0, 0]
        else:
            dstH, dstW = 300, 300
            padH, padW = [0, 0], [0, 0]
        
        tensor = F.interpolate(tensor, (dstH, dstW), mode="bilinear")
        tensor = F.pad(tensor, (padW[0], padW[1], padH[0], padH[1]), "constant", 0)
        H0, W0, scale = padH[0], padW[0], dstH / H

        if self.fp16:
            tensor = tensor.half()
        return tensor, H0, W0, scale
    
    def __call__(self, tensor, class_names=["person"], topk=1):
        """
        tensor should be [bz, 3, H, W] in [-1, 1], RGB
        """
        assert tensor.max() <= 1 and tensor.min() >= -1, \
            "tensor should be [bz, 3, H, W] in [-1, 1], RGB"
        tensor = tensor.to(self.device)
        with torch.no_grad():
            # to 300 x 300
            input, H0, W0, scale = self.prepare_input(tensor)
            
            # torchvision.utils.save_image(
            #     input, "../data/det_in_process.jpg", normalize=True, range=(-1, 1))
            
            predictions = self.model(input)

            ploc, plabel = [val.float() for val in predictions]
            bboxes, probs = self.encoder.scale_back_batch(ploc, plabel)

            if class_names is not None and len(class_names) > 0:
                class_ids = [self.classes_to_labels.index(name) + 1 for name in class_names]
            else:
                class_ids = range(1, len(self.classes_to_labels)+1)

            topk_probs, topk_idxs = probs[:, :, class_ids].topk(topk, dim=1)
            topk_idxs = topk_idxs.view(
                topk_idxs.size(0), -1).unsqueeze(2).repeat(1, 1, 4)
            topk_bboxes = torch.gather(bboxes, dim=1, index=topk_idxs)
            topk_bboxes = topk_bboxes.view(*topk_probs.size(), 4)

            # restore bbox
            topk_bboxes = topk_bboxes * 300
            topk_bboxes -= torch.tensor([W0, H0, W0, H0], device=self.device).float()
            topk_bboxes /= scale

            return topk_bboxes, topk_probs    

    def load_images(self, image_files):
        """
        return tensor with [bz, 3, H, W] in [-1, 1], RGB
        """
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        inputs = []
        for image_file in image_files:
            image = Image.open(image_file).convert("RGB")
            input = to_tensor(image)
            inputs.append(input)

            # image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            
            # input = np.float32(image)
            # input = (input / 255.0 - 0.5) / 0.5 # TO [-1.0, 1.0]
            # input = input.transpose(2, 0, 1) # TO [3 x H x W]
            # input = torch.from_numpy(input).float()
            # inputs.append(input)
        
        inputs = torch.stack(inputs, dim=0).to(self.device)
        return inputs

