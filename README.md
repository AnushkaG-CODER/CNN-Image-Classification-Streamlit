# CNN Image Classification and Streamlit Application

**AIML Minor Project**

## Project Overview

This project demonstrates handwritten digit classification using a Convolutional Neural Network (CNN) and the development of an interactive web application using Streamlit.

The project is divided into two parts:

- **Part A – CNN Image Classification:** A CNN model is trained using the MNIST dataset to recognize handwritten digits from 0 to 9.
- **Part B – Streamlit Application:** A Student Marks Calculator is developed using Streamlit to calculate total marks, percentage and examination results.

## Objectives

- To understand the working of Convolutional Neural Networks.
- To load and preprocess image datasets.
- To build and train a CNN model for image classification.
- To evaluate the model using test data.
- To visualize training and validation accuracy.
- To develop an interactive application using Streamlit.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Streamlit | Web application development |

## Part A: CNN Image Classification

### Dataset

The project uses the **MNIST Handwritten Digit Dataset**.

- Total training images: 60,000
- Total testing images: 10,000
- Image size: 28 × 28 pixels
- Image type: Grayscale
- Number of classes: 10 (digits 0–9)

The dataset is loaded directly using TensorFlow Keras.

### Data Preprocessing

The following preprocessing steps are performed:

1. Load the MNIST dataset.
2. Reshape the images to include the grayscale channel.
3. Normalize pixel values by dividing them by 255.
4. Prepare the data for CNN training.

### CNN Model Architecture

The Sequential CNN model contains:

1. Input Layer – Accepts images of size 28 × 28 × 1.
2. Convolution Layer – 32 filters with ReLU activation.
3. Max Pooling Layer – Reduces image dimensions.
4. Convolution Layer – 64 filters with ReLU activation.
5. Max Pooling Layer – Further reduces image dimensions.
6. Flatten Layer – Converts feature maps into a one-dimensional array.
7. Dense Layer – Contains 64 neurons with ReLU activation.
8. Output Layer – Contains 10 neurons with Softmax activation.

### Model Training

The model is compiled using:

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Evaluation Metric: Accuracy
- Number of Epochs: 3

### Model Performance

The model achieved the following results during execution:

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | Approximately 98.90% |
| Test Loss | Approximately 0.0339 |
| Correct Predictions | 9,890 |
| Incorrect Predictions | 110 |

The model was evaluated using 10,000 test images.

## Part B: Streamlit Application

### Student Marks Calculator

A simple interactive web application is developed using Streamlit.

### Features

- Accepts the student's name.
- Accepts marks for five subjects.
- Allows marks between 0 and 100.
- Calculates total marks out of 500.
- Calculates the percentage.
- Displays PASS or FAIL.

### Result Calculation

**Total Marks:**

Total = Subject 1 + Subject 2 + Subject 3 + Subject 4 + Subject 5

**Percentage:**

Percentage = Total Marks / 5

**Result Condition:**

- PASS: If marks in all five subjects are at least 35.
- FAIL: If marks in any subject are below 35.

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/CNN-Image-Classification-Streamlit.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Open the Project Folder

```bash
cd CNN-Image-Classification-Streamlit
```

### 3. Install Required Libraries

Make sure Python 3.12 is installed.

```bash
py -3.12 -m pip install tensorflow streamlit matplotlib numpy
```

### 4. Run Part A – CNN Classification

```bash
py -3.12 project_6.py --cnn
```

This command loads the MNIST dataset, trains the CNN model, evaluates its performance and displays the graphs and predictions.

### 5. Run Part B – Streamlit Application

```bash
py -3.12 -m streamlit run project_6.py
```

After running the command, open the local URL displayed in the terminal (usually `http://localhost:8501`) in your browser.

## Project Structure

```text
CNN-Image-Classification-Streamlit/
│
├── project_6.py
├── CNN_Streamlit_Project_Report.pdf
└── README.md
```

## Conclusion

This project demonstrates the practical implementation of CNN-based image classification and the development of a web application using Streamlit.

The CNN model successfully classifies handwritten digits from the MNIST dataset with approximately 98.90% test accuracy. The Streamlit application provides a simple interface for calculating student marks and displaying examination results.

Through this project, I gained practical knowledge of image preprocessing, CNN architecture, model training, model evaluation, data visualization and web application development.

## Author

**Anushka Kiran Gundale**  
**Batch / Class:** October (B)

---

*This project was developed as part of an AIML Minor Project.*
