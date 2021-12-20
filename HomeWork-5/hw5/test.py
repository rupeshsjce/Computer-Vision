# import tensorflow as tf
import os
import os.path as osp
import time
import pickle
import torchvision
import torch
from main import testloader

# from data_util import *
from models import *
from main import run_single_step
from models import ResNet18

def test(PATH, model_instance, testloader):
    # now load the model and test it.
    net = model_instance()
    net.load_state_dict(torch.load(PATH))


    ### Let us look at how the network performs on the whole dataset.
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    overall_predicted = []
    overall_grounds = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            overall_grounds = overall_grounds + list(labels.cpu().detach().numpy())
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            overall_predicted = overall_predicted + list(predicted.cpu().detach().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
		ind_exp_results = {
			"loss": [],
			"tp": [],
			"fp": [],
			"fn": [],
			"pixel_level_iou": [],
			"pred": [],
			"probs": []
		}

		exp_results = {}
		for key in ind_exp_results:
			exp_results[key] = np.stack(ind_exp_results[key])
		exp_results["loss"] = np.mean(exp_results["loss"])
		exp_results["tp"] = np.sum(exp_results["tp"])
		exp_results["fp"] = np.sum(exp_results["fp"])
		exp_results["fn"] = np.sum(exp_results["fn"])
		exp_results["pixel_level_iou"] = exp_results["tp"] / (
				exp_results["tp"] + exp_results["fp"] + exp_results["fn"])
		print("Test Average Loss: {:3f}; Test IOU: {:3f}".format(
				exp_results["loss"], exp_results["pixel_level_iou"]))


		
if __name__ == '__main__':
    model_download = torchvision.models.resnet18(pretrained=True)
    model_instance = ResNet18(model_download) # this is fcn32 with resnet18 as backbone
    PATH = './fcn32_net.pth'
    test(PATH, model_instance, testloader)
