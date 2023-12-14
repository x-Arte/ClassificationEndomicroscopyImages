import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from meat import get_trained_model
from vgg_pretrained import get_vgg19_model
from EndomicroscopyDataset import get_transform


class SaveFeatures:
    """ get features & grad"""


    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.hook_backward = module.register_backward_hook(self.hook_backward_fn)
        self.features = None
        self.gradients = None

    def hook_fn(self, module, input, output):
        print("Forward hook called")
        self.features = output.detach()

    def hook_backward_fn(self, module, grad_in, grad_out):
        print("Backward hook called")
        self.gradients = grad_out[0].detach()

    def close(self):
        self.hook.remove()
        self.hook_backward.remove()
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


def grad_cam(model, feature_module, image_tensor, target_category):
    model.eval()
    sf = SaveFeatures(feature_module)
    output = model(image_tensor)
    if output.ndim != 2 or output.shape[0] != 1:
        raise ValueError("Unexpected output shape from the model")

    if output.ndim == 2:
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_category] = 1
        model.zero_grad()
        output.backward(gradient=one_hot_output, retain_graph=True)
    else:
        raise RuntimeError("Model output is not multi-class.")

        # check if get the grad
    if sf.gradients is None:
        raise AttributeError("No gradients captured - backward hook not triggered.")

    # get con & pred
    conf, pred = torch.max(F.softmax(output, dim=1), dim=1)

    one_hot_output = torch.zeros_like(output)
    one_hot_output[0][target_category] = 1

    model.zero_grad()
    output.backward(gradient=one_hot_output, retain_graph=True)


    gradients = sf.gradients.cpu().data.numpy()[0]
    features = sf.features.cpu().data.numpy()[0]
    # calculate weights
    weights = np.mean(gradients, axis=(1, 2))
    #
    cam = np.zeros(features.shape[1:], dtype=np.float32)  # 初始化CAM为零
    for i, w in enumerate(weights):
        cam += w * features[i, :, :]

    # ReLU
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    sf.close()
    return cam, conf.item(), pred.item()


def visualize_cam(data_path, model, final_conv):
    eimg = torch.load(data_path)
    transform = get_transform('vgg')
    img = transform(eimg.image)
    img = torch.concat((img, img, img), dim=0)
    img = img.unsqueeze(0)  # add dim

    # get Grad-CAM
    cam, confidence, prediction = grad_cam(model, final_conv, img, eimg.label)
    cam_img = np.uint8(255 * cam)
    print(eimg.image.shape)
    height, width = eimg.image.shape
    cam_img = np.uint8(Image.fromarray(cam_img).resize((width, height)))

    heatmap = plt.get_cmap('jet')(cam_img)[:, :, :3] * 255
    gray_image_3channel = np.stack((eimg.image,) * 3, axis=-1)
    # ensure heatmap eimg,image have same shape
    cam_img = heatmap * 0.3 + gray_image_3channel * 0.5
    plt.imshow(np.uint8(cam_img))
    plt.title(data_path[data_path.rfind('/')+1:data_path.rfind('.')])
    plt.show()

if __name__ == "__main__":
    modelpath = "model/2023-12-12-18-04.pt"
    ## replace ##
    data_path = 'meat/dataset/test/ck1275_135.pt'
    # meat
    model = get_trained_model("model/2023-12-12-18-04.pt", 3, 64)
    model.load_state_dict(torch.load("meat/2023-12-13-16-44.pt"))
    final_conv = model.features[34]
    for i in model.parameters():
        i.requires_grad = True
    visualize_cam(data_path, model, final_conv)