# Bodhitree AI Server
A Django-based server that handles data collection and grading requests for the Bodhitree platform. The project consists of two main applications: Datacollector and TABuddy, which handle data storage and grading tasks respectively.
## Project Structure
Bodhitree_AI_Server/

    ├── .dvc/                      # DVC (Data Version Control) configuration
    ├── Bodhitree_AI_Server/       # Main Django project directory
    │   ├── __init__.py
    │   ├── asgi.py               # ASGI configuration
    │   ├── celery.py            # Celery configuration for async tasks
    │   ├── settings.py          # Django settings
    │   ├── urls.py              # Main URL configuration
    │   └── wsgi.py              # WSGI configuration
    ├── Datacollector/            # App for data collection and retraining
    │   ├── services/            # Retraining and evaluation modules
    │   ├── utils/               # Utility functions and resources
    │   │   ├── prompts/        # System and task prompts
    │   │   ├── retraining_datasets/  # Dataset files (train, test, eval)
    │   │   └── finetuning_models/    # Previous finetuned models
    │   ├── __init__.py
    │   ├── admin.py
    │   ├── apps.py
    │   ├── exp.py
    │   ├── models.py
    │   ├── tasks.py            # Asynchronous task definitions
    │   ├── tests.py
    │   ├── urls.py
    │   └── views.py
    ├── TABuddy/                 # App for grading requests
    │   ├── services/           # Inference task related scripts
    │   ├── utils/              # Inference models and system prompts
    │   ├── __init__.py
    │   ├── admin.py
    │   ├── apps.py
    │   ├── models.py
    │   ├── serializers.py
    │   ├── tasks.py           # Async query processing tasks
    │   ├── tests.py
    │   ├── urls.py
    │   └── views.py
    ├── retraining_models/      # Storage for retrained models
    ├── .dvcignore
    ├── .gitignore
    ├── Dockerfile
    ├── Dockerfile.mlflow
    ├── docker-compose.yml
    ├── environment.yml
    ├── manage.py
    └── packages_list.txt

## Applications
### Datacollector

- Handles data storage from the Bodhitree server
- Manages asynchronous retraining tasks
- Contains evaluation modules and utilities
- Manages datasets for training, testing, and evaluation
- Maintains previous finetuned models

## TABuddy

- Processes grading requests from the Bodhitree server
- Executes asynchronous query processing tasks
- Contains inference-related scripts and utilities
- Manages inference models and system prompts

## Setup and Installation

Clone the repository
''' git clone <repository-URL>
    cd Bodhitree_AI_Server
'''
Install dependencies:

'''
conda env create -f environment.yml
conda activate bodhitree-ai-server
'''

Set up environment variables
Run migrations:
'''
python manage.py migrate
'''

## Docker Support
The project includes Docker configuration for containerized deployment:

### Dockerfile: Main application container
### Dockerfile.mlflow: MLflow tracking server
### docker-compose.yml: Multi-container orchestration

## Data Version Control
The project uses DVC for managing large files and datasets. The .dvc directory contains the necessary configuration.
Async Task Processing
Both applications use Celery for handling asynchronous tasks:

## Datacollector: Retraining tasks
## TABuddy: Query processing and grading tasks

## Project Dependencies
All required packages are listed in:

- environment.yml: Conda environment specification
- packages_list.txt: Additional Python packages

## Model Management

- New retrained models are stored in the retraining_models directory
- Previous finetuned models are kept in Datacollector/utils/finetuning_models
- Inference models are stored in TABuddy/utils