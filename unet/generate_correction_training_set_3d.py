import SimpleITK
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from utils import swap_labels
#import winshell

############################### CONFIGURATION #####################################
DEBUG = 0 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1

############################## REGISTRATION PARAMETER FILES ######################
similarityMetricForReg = 'NMI'
parameterFilesPath = '..\\..\\Data\\Elastix\\'
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg
paramFileNonRigid = 'Parameters_BSpline_NCC_1000iters_2048samples'#Par0000bspline_500'
############################## AUGMENTATION PARAMETERS ##########################
rotationValues_deg = range(-10, 10+1, 5)
############################### IMAGES AVAILABLE ###################################
#atlasNamesImplantOrNotGood = ['7286867', '7291398', '7300327', '7393917', 'L0469978', 'L0483818', 'L0508687', 'L0554842']
#dataPath = '..\\..\\Data\\Pelvis3D\\Fusion\\' # Base data path.
dataPath = 'C:\\Users\\ecyt\\Documents\\Flor Sarmiento\\CNN correction\\MuscleSegmentation\\Data\\Pelvis3D\\Fusion\\' # Base data path.
outputPath = 'C:\\Users\\ecyt\\Documents\\Flor Sarmiento\\CNN correction\\MuscleSegmentation\\Data\\Pelvis3D\\FusionTrainingSet\\' # Base data path.
outputAugmentedLinearPath = '..\\..\\Data\\Pelvis3D\\TrainingSetAugmentedLinear\\' # Base data path.
outputAugmentedNonLinearPath = '..\\..\\Data\\Pelvis3D\\TrainingSetAugmentedNonLinear\\' # Base data path.
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
if not os.path.exists(outputAugmentedLinearPath):
    os.makedirs(outputAugmentedLinearPath)
if not os.path.exists(outputAugmentedNonLinearPath):
    os.makedirs(outputAugmentedNonLinearPath)
# Get the atlases names and files:
# Look for the folders or shortcuts:
data = os.listdir(dataPath)
# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagInPhase = '_I'
tagAutLabels = '_aut'
tagManLabels = '_manual'
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images
atlasManLabelsFilenames = [] # Filenames of the label images
atlasAutLabelsFilenames = []
folderIndex = []
#for folder in data:
#    auxPath = dataPath + folder + '\\'
#    files = os.listdir(auxPath)
for filename in data:
    name, extension = os.path.splitext(filename)
    # Substract the tagInPhase:
    atlasName = name[:-len(tagInPhase)]

    # Check if filename is the in phase header and the labels exists:
    filenameImages = dataPath + atlasName + tagInPhase + '.' + extensionImages
    filenameAutLabels = dataPath + atlasName + tagAutLabels + '.' + extensionImages
    filenameManLabels = dataPath + atlasName + tagManLabels + '.' + extensionImages
    if name.endswith(tagInPhase) and extension.endswith(extensionImages) and os.path.exists(filenameAutLabels):
        #\ and (atlasName not in atlasNamesImplantOrNotGood):
        # Atlas name:
        atlasNames.append(atlasName)
        # Intensity image:
        atlasImageFilenames.append(filenameImages)
        # Automatic Labels image:
        atlasAutLabelsFilenames.append(filenameAutLabels)
        # Manual Labels image:
        atlasManLabelsFilenames.append(filenameManLabels)

print("Number of atlases images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))


################################### REFERENCE IMAGE FOR THE REGISTRATION #######################
indexReference = 1
referenceSliceImage = sitk.ReadImage(atlasImageFilenames[indexReference])    #Es una Reference image no un Reference slice image
#referenceSliceImage = referenceSliceImage[:, :, 0]
print('Reference image: {0}. Voxel size: {1}'.format(atlasImageFilenames[indexReference], referenceSliceImage.GetSize()))

################################### READ IMAGES, EXTRACT SLICES AND REGISTER IMAGES TO THE REFERENCE ########################################
for i in range(0, len(atlasNames)):
    print('Altas:{0}\n'.format(atlasImageFilenames[i]))
    ############## 1) READ IMAGE WITH LABELS #############     #poner 3
    # Read target image:
    atlasSliceImage = sitk.ReadImage(atlasImageFilenames[i])
    atlasSliceAutLabel = sitk.ReadImage(atlasAutLabelsFilenames[i])
    atlasSliceManLabel = sitk.ReadImage(atlasManLabelsFilenames[i])

    #atlasSliceImage = atlasSliceImage[:, :, 0]
    #atlasSliceLabel = atlasSliceLabel[:, :, 0]
    # Cast the image as float:
    atlasSliceImage = sitk.Cast(atlasSliceImage, sitk.sitkFloat32)   #lo convierte en float 32
    # Rigid registration to match voxel size and FOV.
    ############## 1) RIGID REGISTRATION #############
    # elastixImageFilter filter
    elastixImageFilter = sitk.ElastixImageFilter()
    # Parameter maps:
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath + paramFileRigid + '.txt'))
    # Registration:
    elastixImageFilter.SetFixedImage(referenceSliceImage)
    elastixImageFilter.SetMovingImage(atlasSliceImage)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    # Get result and apply transform to labels:
    # Get the images:
    atlasSliceImage = elastixImageFilter.GetResultImage()

    # Apply transform:
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetMovingImage(atlasSliceAutLabel)
    transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
    transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
    transformixImageFilter.Execute()
    transformixImageFilter.SetMovingImage(atlasSliceAutLabel)   #se la tiene que aplicar a ambos --> aut
    transformixImageFilter.SetMovingImage(atlasSliceManLabel)
    transformixImageFilter.Execute()
    atlasSliceAutLabel = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
    atlasSliceManLabel = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)   #se la tiene que aplicar a ambos --> aut
    # write the 3d images:
    sitk.WriteImage(atlasSliceImage, outputPath + atlasNames[i] + '.' + extensionImages)
    sitk.WriteImage(atlasSliceAutLabel, outputPath + atlasNames[i] + tagAutLabels + '.' + extensionImages)
    sitk.WriteImage(atlasSliceManLabel, outputPath + atlasNames[i] + tagManLabels + '.' + extensionImages)
    # Show images:
    if DEBUG:
        slice = sitk.GetArrayFromImage(atlasSliceImage)
        autLabels = sitk.GetArrayFromImage(atlasSliceAutLabel)
        manLabels = sitk.GetArrayFromImage(atlasSliceManLabel)
        plt.subplot(1,3,1)
        plt.imshow(slice, cmap='gray')
        plt.imshow(autLabels, cmap='hot', alpha=0.5)

    ################################### AUGMENTATE WITH REFLECTION AND ROTATION ########################################
    for reflectionX in [-1,1]:
        ############## Reflection ######################
        imageArray = sitk.GetArrayFromImage(referenceSliceImage)
        imageCenter_mm = np.array(referenceSliceImage.GetSpacing()) * np.array(referenceSliceImage.GetSize())/2; #0.5 * len(imageArray), 0.5 * len(imageArray[0])]
        scale = SimpleITK.ScaleTransform(3, (reflectionX, 1, 1))   #chequear si quiero q refleje en x #A 2D or 3D anisotropic scale of coordinate space around a fixed center.
        scale.SetCenter(imageCenter_mm)
        #if reflectionX == -1:
        #    filter = sitk.FlipImageFilter()
        #    filter.SetFlipAxes((True, False))
        #    atlasSliceImageTransformed = filter.Execute(atlasSliceImage)
        #    atlasSliceLabelTransformed = filter.Execute(atlasSliceLabel)
        #else:
        #    atlasSliceImageTransformed = atlasSliceImage
        #    atlasSliceLabelTransformed = atlasSliceLabel
        for rotAngle_deg in rotationValues_deg: #rotationValues_deg (definidos antes)= range(-10, 10+1, 5)
            rotation3D = sitk.Euler3DTransform()
            #rotation2D.SetAngle(np.deg2rad(rotAngle_deg))
            rotation3D.SetRotation(0, 0, np.deg2rad(rotAngle_deg))
            rotation3D.SetCenter(imageCenter_mm)
            # Composite transform: (junta las dos transofrmadas)
            composite = sitk.Transform(scale)
            composite.AddTransform(rotation3D)
            #scale.SetScale((-1,1))
            # Apply transform:
            atlasSliceImageTransformed = sitk.Resample(atlasSliceImage, composite, sitk.sitkLinear, 0)
            atlasSliceAutLabelTransformed = sitk.Resample(atlasSliceAutLabel, composite, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            atlasSliceManLabelTransformed = sitk.Resample(atlasSliceManLabel, composite, sitk.sitkNearestNeighbor, 0,
                                                       sitk.sitkUInt8)
            # Change the labels side:
            if reflectionX == -1:
                for l in range(1, 6, 2):
                    atlasSliceAutLabelTransformed = swap_labels(atlasSliceAutLabelTransformed, label1=l, label2=l+1)
                    atlasSliceManLabelTransformed = swap_labels(atlasSliceManLabelTransformed, label1=l, label2=l + 1)
            # write the 2d images:
            sitk.WriteImage(atlasSliceImageTransformed, outputAugmentedLinearPath + atlasNames[i] + '_refX' + str(reflectionX) + '_rotDeg' + str(rotAngle_deg) +'.' + extensionImages)
            sitk.WriteImage(atlasSliceAutLabelTransformed, outputAugmentedLinearPath + atlasNames[i] + '_refX' + str(reflectionX) + '_rotDeg' + str(rotAngle_deg) +  tagAutLabels  + '.' + extensionImages)
            sitk.WriteImage(atlasSliceManLabelTransformed, outputAugmentedLinearPath + atlasNames[i] + '_refX' + str(reflectionX) + '_rotDeg' + str(rotAngle_deg) + tagManLabels + '.' + extensionImages)

            # Show images:
            if DEBUG:
                slice = sitk.GetArrayFromImage(atlasSliceImageTransformed)
                autLabels = sitk.GetArrayFromImage(atlasSliceAutLabelTransformed)   #aca deberia guardar ambas?
                manLabels = sitk.GetArrayFromImage(atlasSliceManLabelTransformed)
                plt.subplot(1, 3, 3)  #aca algo tengo que cambiar
                plt.imshow(slice, cmap='gray')
                plt.imshow(autLabels, cmap='hot', alpha=0.5)
                plt.imshow(manLabels, cmap='cold', alpha=0.5)
                plt.show()

    ################################### AUGMENTATE WITH NONLINEAR TRANSFORMATIONS ########################################
    for j in range(0, len(atlasNames)):
        # Image to realign to:
        fixedSliceImage = sitk.ReadImage(atlasImageFilenames[j])
        #fixedSliceImage = fixedSliceImage[:, :, 0]
        ############## NONRIGID REGISTRATION #############
        # elastixImageFilter filter
        elastixImageFilter = sitk.ElastixImageFilter()
        # Parameter maps:
        parameterMapVector = sitk.VectorOfParameterMap()
        parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                       + paramFileRigid + '.txt'))
        parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                       + paramFileNonRigid + '.txt'))
        # Registration:
        elastixImageFilter.SetFixedImage(fixedSliceImage)
        elastixImageFilter.SetMovingImage(atlasSliceImage)
        elastixImageFilter.SetParameterMap(parameterMapVector)
        elastixImageFilter.Execute()
        # Get result and apply transform to labels:
        # Get the images:
        atlasSliceImageDeformed = elastixImageFilter.GetResultImage()

        # Apply transform:
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetMovingImage(atlasSliceAutLabel)
        transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
        transformixImageFilter.Execute()
        atlasSliceAutLabelDeformed = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
        transformixImageFilter.SetMovingImage(atlasSliceManLabel)
        transformixImageFilter.Execute()
        atlasSliceManLabelDeformed = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
        # write the 2d images:
        sitk.WriteImage(atlasSliceImageDeformed, outputAugmentedNonLinearPath + atlasNames[i] + '_' + atlasNames[j] + '.' + extensionImages)
        sitk.WriteImage(atlasSliceAutLabelDeformed, outputAugmentedNonLinearPath + atlasNames[i] + '_' + atlasNames[j] + tagAutLabels + '.' + extensionImages)
        sitk.WriteImage(atlasSliceManLabelDeformed, outputAugmentedNonLinearPath + atlasNames[i] + '_' + atlasNames[j] + tagManLabels + '.' + extensionImages)

        # Show images:
        # slice = sitk.GetArrayFromImage(atlasSliceImageDeformed)
        # labels = sitk.GetArrayFromImage(atlasSliceLabelDeformed)
        # plt.subplot(1, 2, 2)
        # plt.imshow(slice, cmap='gray')
        # plt.imshow(labels, cmap='hot', alpha=0.5)
        # plt.show()


