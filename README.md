# Brain Tumor Detection using CNN

## Introduction
Brain tumors are life-threatening, and detecting them early is crucial for effective treatment. However, detecting brain tumors and identifying their type requires highly skilled professionals and can be time-consuming and costly. This project aims to aid doctors by providing a deep learning-based solution to detect brain tumors and their types from MRI images, thus reducing the time and cost involved in diagnosis.

## Dataset
The dataset used in this project is publicly available on GitHub and contains over 2000 MRI images of the brain. To prepare the data for model training, several preprocessing steps were performed, including resizing the images, normalization, and more. Link to the dataset can be found [here](https://github.com/sartajbhuvaji/brain-tumor-classification-dataset)

## Model Architecture
The Convolutional Neural Network (CNN) model developed for this project is composed of 5 layers:
- **4 Convolutional Layers**: These layers help detect different features in the images by applying filters. Each layer utilizes max pooling and dropout to improve performance and reduce overfitting.
- **1 Dense Layer**: This layer has 128 neurons and uses the ReLU activation function to introduce non-linearity, enabling the model to learn complex patterns.
- **Output Layer**: Connected to the dense layer, this layer provides the final classification output.

The model employs the Adam optimizer (Adaptive Moment Estimation) to enhance the learning process by adjusting the learning rate during training.

## Training
The dataset was split into training, validation, and testing sets, with the majority allocated to training. The model was trained over 45 epochs with a batch size of 64 and a learning rate of 0.001.

## Evaluation
The model achieved an accuracy of 97% on the training data and 93% on the validation data. The F1 score, a crucial metric for imbalanced datasets, was 0.93, indicating a strong performance in tumor detection and classification.

## Results and Visualization
Results were visualized using a confusion matrix, which provides a clear depiction of the model’s performance across different classes. Additionally, the model includes functionality to visualize predictions on sample inputs, helping users understand how the model makes decisions.

## Usage
This project is intended to assist doctors in diagnosing brain tumors, potentially saving time and resources. To use the model, simply upload an MRI image, and the model will output whether a tumor is present and its type.

## Conclusion
This project demonstrates the potential of deep learning in medical diagnosis. While the current model performs well, it can be further improved by training on larger datasets to expose the model to a wider variety of tumor locations within the brain. Additionally, more labels could be added to detect various other conditions, such as hematomas, hemorrhages, and more. Various small improvements to the model architecture and hyperparameters could also enhance performance.

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to the creators of the publicly available dataset and to the open-source community for providing the tools and resources that made this project possible.
