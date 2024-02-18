import SimpleITK as sitk
import numpy as np
import os
from unet_3d import Unet
import torch
from utils import multilabel
from utils import maxProb
from utils import FilterUnconnectedRegions
from utils import cuda_memoryUsage

############################ DATA PATHS ##############################################
dataPath = 'D:/UnetLumbarSpine/Data/LumbarSpine3D/InputImages/'
outputPath = 'D:/UnetLumbarSpine/Data/LumbarSpine3D/InputImages/'
modelLocation = 'D:/UnetLumbarSpine/Data/LumbarSpine3D/PretrainedModel/'
# Image format extension:
extensionImages = 'mhd'
loadDict = False

if not os.path.exists(dataPath):
    os.makedirs(dataPath)

modelName = os.listdir(modelLocation)[0]
modelFilename = modelLocation + modelName

######################### CHECK DEVICE ######################
device = torch.device('cuda')
print(device)
if device.type == 'cuda':
    cuda_memoryUsage()

######################### MODEL INIT ######################
multilabelNum = 8
if loadDict:
    torch.cuda.empty_cache()
    model = Unet(1, multilabelNum)
    model.load_state_dict(torch.load(modelFilename, map_location=device))
    model = model.to(device)
else:
    model = torch.load(modelFilename,map_location=device)
    model = model.to(device)

trainable_params = [p for p in model.parameters() if p.requires_grad]
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_trainable_params}")
model.eval()

###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #####################################################
# Look for the folders or shortcuts:
files = os.listdir(dataPath)
files = sorted(files)
imageNames = []
imageFilenames = []
i = 0
for filename in files:
    name, extension = os.path.splitext(filename)
    if extension.endswith('raw') or not name.endswith('i'):
        continue

    filenameImage = dataPath + filename
    sitkImage = sitk.ReadImage(filenameImage)
    image = sitk.GetArrayFromImage(sitkImage).astype(np.float32)
    image = np.expand_dims(image, axis=0)

    with torch.no_grad():
        input = torch.from_numpy(image).to(device)
        output = model(input.unsqueeze(0))
        cuda_memoryUsage()
        output = torch.sigmoid(output.cpu().to(torch.float32))
        outputs = maxProb(output, multilabelNum)
        output = ((output > 0.5) * 1)
        output = multilabel(output.detach().numpy())
    output = FilterUnconnectedRegions(output.squeeze(0), multilabelNum, sitkImage,[0,0,0])# Herramienta de filtrado de imagenes
    print(name)

    sitk.WriteImage(output, dataPath + name[0:-2] + '_seg' + extension)
    #writeMhd(output.squeeze(0).astype(np.uint8), outputPath + name + '_segmentation' + extension, sitkImage) # sin herramienta de filtrado