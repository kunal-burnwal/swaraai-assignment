# SvaraAI - Reply Classification Project

This repository contains the solution for the SvaraAI AI/ML Engineer Internship Assignment. The project involves building an end-to-end reply classification pipeline, deploying it as a REST API, and creating a simple visual interface for testing.

### Project Deliverables

-   `pipeline.py`: Python script for the ML/NLP pipeline (Part A), which handles data processing, model training, and serialization.
-   `app.py`: The Flask web server for the prediction API (Part B), which serves the model and the visualizer.
-   `answers.md`: Markdown file containing the reasoning answers (Part C).
-   `requirements.txt`: Lists all necessary Python dependencies.
-   `reply_classification_dataset.csv`: The dataset used for training the model.
-   `templates/index.html`: The HTML file for the simple visualizer.

### Setup and Running Instructions

Follow these steps to set up and run the project locally.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/svaraai-assignment](https://github.com/your-username/svaraai-assignment)
    cd svaraai-assignment
    ```
    *(Note: Replace the URL with your actual GitHub repository URL.)*

2.  **Install Dependencies**
    First, ensure you have Python and `pip` installed. Then, install the required libraries using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Model (Part A)**
    This script will process the dataset, train a Logistic Regression model, and save the necessary model and vectorizer files (`model.joblib` and `vectorizer.joblib`) for the API.
    ```bash
    python pipeline.py
    ```

4.  **Run the Deployment API (Part B)**
    Once the model files are created, you can start the Flask web server.
    ```bash
    python app.py
    ```
    The API and the visualizer will be available at `http://127.0.0.1:5000`.

### Visualizer and API Endpoints

-   **Visualizer URL**: `http://127.0.0.1:5000/`
    Open this URL in your web browser to access a simple, user-friendly interface for classifying email replies.
-   **API Endpoint**: `/predict`
    -   **Method**: `POST`
    -   **Request Body (JSON)**:
        ```json
        {
          "text": "Your email reply text here."
        }
        ```
    -   **Response Body (JSON)**:
        ```json
        {
          "label": "predicted_label",
          "confidence": 0.95
        }
        ```