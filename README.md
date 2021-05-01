<h1 align ='center'><b>Exploratory-Project</b></h1>
<h2 align='center'>Person-Identification</h2>

[Project proposal](https://drive.google.com/file/d/1d4v-FjKnAP_lMCKRN_KP8ASqq-x333RY/view?usp=sharing)


<h3 align ='center'>Our Appraoch</h3>
<p>
We used a pretrained yolo to extract all the possible persons in an input image, then using MTCNN extracted the faces which will pass via a siamese facenet to match the person with the subjects present in gallery database.
</p>

**To download weights**

```bash
cd yolo_cnfgs
wget https://pjreddie.com/media/files/yolov3.weights
```
**For requirements**
```bash
pip install opencv-python numpy
pip install facenet-pytorch
```

[Report](Exploratory Report (19095009-12).pdf)
[Final-Presentation](https://docs.google.com/presentation/d/1Mb5L-_ywcULafsDxovUbLNHrCh3a8EVWXlwvK3sSymQ/edit?usp=sharing)
## Resources
 - [PyTorch-tutorials](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)
 - [pytorch-official docs](https://pytorch.org/docs/stable/index.html)
 - [KMeans Clustering](https://towardsdatascience.com/k-means-clustering-from-a-to-z-f6242a314e9a)
 - [Siamese Net](https://towardsdatascience.com/building-a-one-shot-learning-network-with-pytorch-d1c3a5fafa4a)
 - [MTCNN-for face detection ](https:mediumcom@iselagradilla94how-to-build-a-face-detection-application-using-pytorch-and-opencv-d46b0866e4d6)