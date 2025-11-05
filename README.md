# RSVDet
RSVDet: Remote Sensing Small Object Detection Model Inspired by Neuronal Mechanisms in Visual Pathways

Small object detection is a signiffcant and challenging research area in aerospace. Small objects often face issues like background interference and inter-class similarity due to their small size and low boundary contrast in complex environments. Physiological studies indicate that the visual pathway’s neuronal
mechanisms can effectively extract features such as contours, shapes, and colors, fflter out background noise and thus recognize complex forms. Therefore, this paper proposes a remote sensing small object detection model inspired by neuronal mechanisms in visual pathways, called RSVDet. RSVDet simulates the information transmission of the ventral visual pathway (Retina-LGNV1-V2-V4-IT) and meticulously models the involved visual areas. First, inspired by the “Retina-LGN-V1” pathway, we designed a feature enhancement module to capture low-level information. Second, based on the global-local receptive ffeld mechanism of V2 neurons, we developed a feature extraction module for shape information. Additionally, inspired by the self-regulation mechanisms of V4 neurons, we created a self-feedback attention module to fflter background noise. Finally, drawing from the orientation selectivity of IT neurons, we designed a hierarchical modulation detection head module to extract complex shape features. RSVDet achieves an AP50 of 50.1% on the Visdrone dataset with 1.72M parameters, achieving the best performance
among lightweight models.
![image](https://github.com/user-attachments/assets/bbfe3719-3d7b-4f8f-8e1f-7961cccd7864)


Cite ：https://ieeexplore.ieee.org/document/11224383
