import matplotlib.pyplot as plt
import matplotlib
import torch
import cv2
from torchvision import transforms
import numpy as np
from util.datasets import letterbox
from util.general import non_max_suppression_kpt
from util.plots import output_to_keypoint, plot_skeleton_kpts
from models.yolo import Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
model = weigths['model']

_ = model.float().eval()
'''
model.half()不能在16系列的显卡中使用,会出现nan,因此需要改为mode.float()
'''

if torch.cuda.is_available():
    # model.half().to(device)
    model.float().to(device)
video = cv2.VideoCapture('F:/save_cut/ABAD Nestor (ESP) - 2015 Artistic Worlds - Qualifications Vault 1.mp4/.mp4001.mp4')



while True:
    cv2.namedWindow('ddd')
    ret,image = video.read()
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        # image = image.half().to(device) 
        image = image.float().to(device)   
    output, _ = model(image)
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)

    with torch.no_grad():
        output = output_to_keypoint(output)

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    key = cv2.waitKey(1)
    if key=='q':
        exit()
    cv2.imshow('ddd',nimg)
cv2.destroyAllWindows()
video.release()  # 释放视频流资源
    

    # plt.figure(figsize=(16,16))
    # plt.axis('off')
    # plt.imshow(nimg)
    # plt.show()