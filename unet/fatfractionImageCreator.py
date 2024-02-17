import SimpleITK as sitk
import numpy as np
import os


dataPath ='D:/Dixon German Balerdi/'
outputPath = 'D:/Dixon German Balerdi/'
folder = sorted(os.listdir(dataPath))

extensionImages = '.mhd'
inPhaseSuffix = '_i'
outOfPhaseSuffix = '_o'
waterSuffix = '_w'
fatSuffix = '_f'

auxName=str
for files in folder:
    name = os.path.splitext(files)[0]
    if name.split('_')[0] != auxName:
        auxName = name.split('_')[0]
        fatImage = sitk.Cast(sitk.ReadImage(dataPath + auxName + fatSuffix + extensionImages), sitk.sitkFloat32)
        waterImage = sitk.Cast(sitk.ReadImage(dataPath + auxName + waterSuffix + extensionImages), sitk.sitkFloat32)
    else:
        continue
    waterfatImage = sitk.Add(fatImage, waterImage)
    fatfractionImage = sitk.Divide(fatImage, waterfatImage)
    fatfractionImage = sitk.Cast(sitk.Mask(fatfractionImage, waterfatImage > 0, outsideValue=0, maskingValue=0), sitk.sitkFloat32)
    sitk.WriteImage(fatfractionImage, outputPath + auxName + '_ff' + extensionImages)