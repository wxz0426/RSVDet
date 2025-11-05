from ultralytics import YOLO
import os
from tqdm import tqdm
import os
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def main():
    model = YOLO(r'ultralytics\cfg\models\RSVDet\RSVDet.yaml')

# Train the model

    model.train(data=r'ultralytics\cfg\datasets\VisDrone.yaml', batch=8, epochs=1000, imgsz=640, workers=0, seed=0, device=0,
            pretrained=False, lr0=0.01,
            resume=True)

if __name__ == '__main__':
    main()

# #######-------------------The test code for parameters and computational complexity---------------------------------------------
# from ultralytics import YOLO
# from thop import profile
# import torch
#
#
# if __name__ == "__main__":
#     model = YOLO(r'D:\MBJC\论文返修\tcsvt最终版\code\DDCNet-main\DDCNet-main\ultralytics\cfg\models\RSVDet\RSVDet.yaml')
#     model.model.eval()
#     # exit(0)
#     input = torch.randn(1, 3, 384, 672)   # 384, 672   VisDrone has an input size of 384 × 672 during validation
#     macs, params = profile(model.model, inputs=(input, ))
#     print('MACs = ' + str(macs/1000**3) + 'G')
#     print('Params = ' + str(params/1000**2) + 'M')