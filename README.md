# Object Detection of Drawings

This is an object detection project developed with Java using the [Deeplearning4j](https://github.com/eclipse/deeplearning4j) library. The project aims to detect drawings of car, cat, and tree categories by showing drawings to the webcam.

![Object detection on test images](https://media4.giphy.com/media/u9RJoI3kytsIL0ZrKk/giphy.gif)
![Object detection on webcam](https://media2.giphy.com/media/qu2sIB6lEcPkh3neUH/giphy.gif)

## Dataset

This project used three classes of image dataset which are cat, car and tree. The dataset has been annotated for testing and training purposes.

Sources:

- Images from the Internet
- Quick, Draw! by Google ([Github](https://github.com/googlecreativelab/quickdraw-dataset))

## Network Description

The Pre-trained Tiny Yolo comprised 9 convolutions layer with 6 max pooling. We fine-tuning the output layer for classification.

## Future Development

This project plans to train with more drawings categories and achieve higher detection accuracy.

## Video

[Youtube](https://www.youtube.com/watch?v=aY2e23eHQ70)

## Group Members

| Name                                                      | Email                  |
| --------------------------------------------------------- | ---------------------- |
| [Annabelle Ng Xin Min](https://github.com/xin133)         | ngxinmin.n@gmail.com   |
| [Lee Jue Min](https://github.com/JueMinLee)               | jmlee.works@gmail.com  |
| [Muhammad Izzuddin Abd Hamid](https://github.com/mizudin) | mizudinhamid@gmail.com |

## Notes

- First time run the program will train the model. Once the trained model is saved, run the program second time to perform object detection.
- Trained model zip file `DrawingObjectDetection_tinyyolo.zip` will be save inside `generated-models` folder, `deleteme` file can be delete.
- When the program is running, press <kbd>Esc</kbd> key to exit program, press any key to go through the images.
