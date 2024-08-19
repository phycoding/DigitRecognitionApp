
# Digit Recognition App 🚀

Welcome to the **Digit Recognition App**! This application allows you to train and test a digit recognition model, as well as interact with it using a Streamlit web interface. Follow the steps below to get everything up and running. 🎉

## 📂 Setup Instructions

### 1. Clone the Repository

Start by cloning this repository to your local machine:

```bash
git clone https://github.com/phycoding/DigitRecognitionApp.git
cd DigitRecognitionApp
```

### 2. Create and Activate Virtual Environment

Create a virtual environment and activate it. This ensures all dependencies are isolated and managed properly. 🧑‍💻

#### On Windows:
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install all required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Train the Model 🏋️

Run the training script to train the digit recognition model:

```bash
cd model
python train.py
```
> This will train the digit recognition model which will achive an accuracy of 99 percent on validation data. It will generate some training logs and display the results.

### 5. Test the Model 🧪

```bash
cd model
python test.py
```
> This will test the model it will generate some confusion metrics, accuracy metrics and more.
After training, test the model to ensure it's performing correctly:


### 6. Run the Streamlit App 🌟

Start the Streamlit application to interact with the digit recognition model:

```bash
streamlit run app.py
```

## 📄 Additional Information

- **Model Training**: The `train.py` script trains the digit recognition model using TensorFlow.
- **Model Testing**: The `test.py` script evaluates the model's performance on test data.
- **Streamlit Application**: The `app.py` file provides a web-based interface for drawing digits and predicting them using the trained model.

## 📸 Screenshots & GIFs

Here are some visuals to help you understand the application better:

- **Training Graph**: ![Training Graph]("images\model_statistics.png)
- **Testing Script**: ![Testing GIF]()
- **Streamlit App**: ![Streamlit App Screenshot]()

## 🎉 Thank You!

>Thank you for using the Digit Recognition App! If you have any questions or need further assistance, feel free to open an issue on GitHub or contact us.
---