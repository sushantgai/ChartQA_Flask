# Flask Chart Analysis & Data Extraction

## Overview
This is a Flask-based web application that allows users to upload chart images, analyze them using the PaliGemma model, and extract structured data points in CSV format. The application leverages deep learning to interpret charts and generate meaningful insights.

## Features
- Upload images of charts (PNG, JPG, JPEG)
- Analyze the chart using PaliGemma model
- Extract data points from the chart and convert them into structured CSV format
- Download extracted data as a CSV file
- Uses Flask for the web framework and PyTorch for deep learning

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)
- PyTorch (with CUDA if GPU is available)
- Required Python dependencies (listed below)

## Installation
### 1. Clone the repository:
```sh
$ git clone https://github.com/your-repo/flask-chart-analysis.git
$ cd flask-chart-analysis
```

### 2. Create a virtual environment:
```sh
$ python -m venv venv
$ source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies:
```sh
$ pip install -r requirements.txt
```

### 4. Download the PaliGemma model:
Ensure that the `Model/` directory contains the necessary model files. If not, download them manually and place them in the correct path.

## Usage
### Running the Flask App
```sh
$ python app.py
```
The application will run on `http://127.0.0.1:5000/`.

### Uploading and Analyzing Charts
1. Open `http://127.0.0.1:5000/` in a browser.
2. Upload an image of a chart.
3. Enter a query and analyze the chart.
4. Extract data points and download the CSV file.

## API Endpoints
### `/upload` (POST)
Uploads an image and stores it in the session.

**Request:**
- `image` (file): Chart image file (PNG, JPG, JPEG)

**Response:**
- `image_url`: URL of the uploaded image

### `/analyze` (POST)
Analyzes the uploaded chart using the PaliGemma model.

**Request:**
- `query` (string): Text query for analysis
- `use_cot` (boolean): Whether to use Chain of Thought (CoT) prompting

**Response:**
- `answer`: Generated response from the model

### `/extract` (POST)
Extracts data points from the uploaded image and returns them as CSV.

**Response:**
- `csv_data` (Base64-encoded CSV)

### `/download_csv` (GET)
Downloads the extracted data as a CSV file.

## Folder Structure
```
flask-chart-analysis/
│-- Model/                 # PaliGemma model files
│-- static/uploads/        # Uploaded images
│-- templates/
│   ├── index.html         # Frontend template
│-- app.py                 # Flask application
│-- requirements.txt       # Dependencies
│-- README.md              # Documentation
```

## Dependencies
- Flask
- PyTorch
- Transformers
- PIL (Pillow)
- Pandas
- Base64

Install dependencies using:
```sh
$ pip install -r requirements.txt
```

## Troubleshooting
### Model Loading Issues
- Ensure the `Model/` directory contains the correct files.
- Verify that PyTorch is installed correctly (`torch.cuda.is_available()` should return `True` if using GPU).

### Image Not Found
- Ensure uploaded images are in `static/uploads/`.
- Try clearing the session and re-uploading the image.



