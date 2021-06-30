# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:32:27 2021

@author: wzh
"""

import torchvision
import torch
from torch.autograd import Variable
import onnx
import torch.onnx
from torchvision import models

def gpuSaveOnnx(input_pt_file, output_onnx_file, fc_num):
    input_name = ['input']
    output_name = ['output']
    input = Variable(torch.randn(1, 3, 224, 224)).cuda()
    
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, fc_num)
    for param in model.parameters():
        param.requires_grad = False
        break
    loaded_model = torch.load(input_pt_file)
    model.load_state_dict(loaded_model)
    model=model.cuda().eval()
    
    torch.onnx.export(model, input, output_onnx_file, input_names=input_name, output_names=output_name, verbose=True)
    input=input.cpu()
    model=model.cpu()
    torch.cuda.empty_cache()
    
def cpuTwoFeature():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    #pthfile = r'/home/joy/Projects/models/emotion/PlainC3AENet.pth'
    loaded_model = torch.load('modelx.pt', map_location='cpu')
    # try:
    #   loaded_model.eval()
    # except AttributeError as error:
    #   print(error)
       
    model.load_state_dict(loaded_model)
    # model = model.to(device)
       
    #data type nchw
    dummy_input1 = torch.randn(1, 3, 224, 224)
    # dummy_input2 = torch.randn(1, 3, 64, 64)
    # dummy_input3 = torch.randn(1, 3, 64, 64)
    input_names = [ "actual_input_1"]
    output_names = [ "output1" ]
    # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
    torch.onnx.export(model, dummy_input1, "model2.onnx", verbose=True, input_names=input_names, output_names=output_names)

def cpuSrcResnet():
    net = models.resnet.resnet50(pretrained=True)
    dummpy_input = torch.randn(1,3,224,224)
    torch.onnx.export(net, dummpy_input, 'resnet50.onnx')    
    # Load the ONNX model
    model = onnx.load("resnet50.onnx")    
    # Check that the IR is well formed
    onnx.checker.check_model(model)    
    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))

if __name__ == "__main__": 
    # print(torch.__version__)
    gpuSaveOnnx('model.pt','resnet50.onnx',2)
