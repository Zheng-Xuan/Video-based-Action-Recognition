import mxnet as mx
import numpy as np
from mxnet import gluon, nd, image
from gluoncv.utils import try_import_cv2
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms import video

cv2 = try_import_cv2()

def inference():

    # Download pretrained model
    print("Downloading pretrained model...")
    model_name = 'i3d_inceptionv1_kinetics400'
    net = get_model(model_name, nclass=400, pretrained=True)
    print(f"model: {model_name} downloaded")

    classes = net.classes

    # Real time detection
    print("Starting capture...")
    vid = cv2.VideoCapture(0)

    while True:
        
        # Ingesting video stream
        frames = []

        for i in range(0, 64, 2):

            ret, frame = vid.read()
            if ret:
                frame = cv2.resize(frame, (640, 360))
                frames.append(frame)
        
        cv2.imshow('frame', frame)

        clip = np.array(frames)
        frame_id_list = range(0, 64, 2)
            
        clip_input = [clip[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]

        # Pre-processing
        transform_fn = video.VideoGroupValTransform(size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        clip_input = transform_fn(clip_input)
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (32, 3, 224, 224))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
        print("Input clip has been preprocessed!")

        # Model prediction
        print("Model prediction...")
        pred = net(nd.array(clip_input))
        
        ind = nd.topk(pred)[0].astype('int')
        class_pred = classes[ind.asscalar()]

        print(f"Action is most likely to be {class_pred}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       

if __name__ == '__main__':
    inference()