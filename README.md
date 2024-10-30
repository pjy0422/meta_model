

## Setup Instructions

### Install Dependencies

Use the following command to install the required Python packages:

```bash
pip install -r requirements.txt
```

### Start the MLflow Server

Before running the code, start the MLflow server with the following command:

```bash
mlflow server --host 127.0.0.1 --port 3060
```

This will set up the MLflow tracking server locally on port 3060.

### Run the Pipeline

Navigate to each relevant directory and run the pipeline script with:

```bash
python pipeline.py
```

Make sure to execute this command in each directory where `pipeline.py` is located.

--- 

This version is more organized, providing clear instructions in each section. Let me know if you need further customization!