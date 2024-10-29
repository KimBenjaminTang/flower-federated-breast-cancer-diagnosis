# Federated breast cancer prediction with Flower
Predict whether the cancer is benign or malignant on the Breast Cancer Wisconsin (Diagnostic) Data Set.

[![Build & Test](https://github.com/KimBenjaminTang/federated-breast-cancer-prediction/actions/workflows/docker-image.yml/badge.svg)](https://github.com/KimBenjaminTang/federated-breast-cancer-prediction/actions/workflows/docker-image.yml)

# Task 1: Design a Federated Learning Training Algorithm on the Breast Cancer Wisconsin Data Set Using the Flower Framework

## Objective:
To assess your ability to design and implement a federated learning solution, simulate a decentralized training environment, and demonstrate your knowledge of basic federated learning concepts

## Task Description:
- Implement a federated learning system using the Flower framework on the Breast
Cancer Wisconsin dataset. The dataset will be split into multiple subsets (clients) that
simulate data being held at different hospitals
- The system should train a simple machine learning model in a federated manner, where
each client trains its local model on its subset of the data, and a central server
aggregates the updates


## Steps:
1. **Data Distribution:**
   - Split the Breast Cancer Wisconsin dataset into 3–5 clients, each representing a
different hospital
2. **Model Training:**
   - Implement a machine learning model using a library of your choice
   - Set up the Flower framework to coordinate the training process across clients
3. **Model Aggregation:**
   - Implement an aggregation mechanism, where the central server aggregates the
model updates from clients to produce a global model

## Submission:
1. Code in a Jupyter notebook or Python scripts, well-commented and organized
2. A short README file explaining of your design choices

<details>
<summary> $${\color{blue}Submission}$$ </summary>

### 1. Code in a Jupyter notebook or Python scripts, well-commented and organized
   - Jupyter Notebook: [federated_breast_cancer.ipynb](https://github.com/KimBenjaminTang/federated-breast-cancer-prediction/blob/main/federated_breast_cancer.ipynb)
### 2. A short README file explaining of your design choices
1. **Data Distribution: Horizontal split randomly of patient data across five institutions**
   - horizontal: more common use case than vertical split
   - random: realistic setting for different institutions, underlying distributions can differ
   - five institutions: test case with maximum institutions, number of institutions can be adjusted in code to fewer
   - data split into 80%/20% split for training & testing, where 80% are then split up across institutions -> each institution splits again into 80/20 for training & validation -> final global model evaluation on 20% withheld data
   - dropping patient id and column with nan values (no relevant information for prediction)
   - no implementation of sampling techniques (oversampling, undersampling) of patient data to keep approach simple to focus on flower & docker instead (but mention possibilities)
   - no implementation of feature engineering & selection to keep appraoch simple to focus on flower & docker instead (but mention possibilities)
2. **Model Training: Training of Neural Network in pytorch with flower architecture**
   - neural network choosen for iterative training in epochs across clients (other approach like svm without native epoch-training less straight forward to implement)
   - 10 epochs with default NN parameters to demonstrate working model and accuracy of 85-95%, but no fine-tuning or parameter/architecture optimization
   - simulated run with flower with client, server & task
3. **Model aggregation: Weighted average based on patient samples at institution**
   - training for one epoch and then gathering weights & weighted averaging before next epoch
   - weighted average to account for varying number of samples at institutions & averaging for simplicity
   - using all clients for training & evaluation instead of sampling only a subset to monitor performance & learn across all data subsets


**Overview of the minimal dataset modification:**

![image](https://github.com/user-attachments/assets/27e89c16-8233-4d2b-9352-bbd36bb95f3e)

**Overview of the horizontal data split:**

![image](https://github.com/user-attachments/assets/bd1561b0-e41c-43e0-99f7-fd947bc06675)

**Overview of the data split for the institutions and final model evaluation:**

![image](https://github.com/user-attachments/assets/aa265a99-174c-44fc-8868-a06d4e4aa44d)

</details>

# Task 2: Simulate a Distributed Environment Using Docker Containers for Clients and Server

## Objective:

To assess your ability to simulate a federated learning environment using Docker to containerize both the client and server applications, allowing reproducible setup

## Task Description:

Set up a Docker-based simulation of the federated learning system you implemented in Task 1. Each client and the central server should run in separate Docker containers, simulating a distributed environment.

## Steps:
1. Dockerize the Clients and Server:
    - Create Docker containers for each client and the central server
2. Container Orchestration:
    - Use Docker Compose to orchestrate the containers, ensuring that the clients and server can communicate with each other. The server should coordinate the training rounds and aggregate the updates from clients
  
## Submission:
1. Dockerfiles for the client and server containers
2. A Docker Compose file to orchestrate the system
3. A README file explaining how to set up, run, and test the Dockerized environment

<details>
   
<summary> $${\color{blue}Submission}$$ </summary>

### 1. Dockerfiles for the client and server containers
   - **Client Dockerfile: [clientapp.Dockerfile](https://github.com/KimBenjaminTang/federated-breast-cancer-prediction/blob/main/compose_setup/clientapp.Dockerfile)**
   - **Server Dockerfile: [superexec.Dockerfile](https://github.com/KimBenjaminTang/federated-breast-cancer-prediction/blob/main/compose_setup/superexec.Dockerfile)**
### 2. A Docker Compose file to orchestrate the system
   - **Docker compose file: [compose.yml](https://github.com/KimBenjaminTang/federated-breast-cancer-prediction/blob/main/compose_setup/compose.yml)**
### 3. A README file explaining how to set up, run, and test the Dockerized environment
   - **Ideally, use the github codespace:**
     1. "pip uninstall flwr" & "pip install flwr" (for some reason it is installed but flwr cli cant be used unless its reinstalled)
     2. install python & jupyter extensions in vscode menu
     3. run all cells in federated_breast_canceer.ipynb
     4. in terminal, go to dir: /compose_setup/
     5. Execute "docker compose up -d": Run docker compose file to spin up/build containers
     6. Execute "flwr run breastcancer local-deployment --stream" to run the flower project with the configuration in pyproject.toml for local-deployment
   - **Locally:** install requirements from requirements.txt & then go to /compose_setup/ in cloned repository to continue with step 5. from above.
   
The github workflow also shows how the docker compose & flower execution runs (first stage build client & superexec container, then test docker compose, then test docker compose with flower)**

**This is an overview of the docker compose setup to run flower for the federated learning breast cancere classification:**
![grafik](https://github.com/user-attachments/assets/fb4ea300-a810-4909-b9b5-07a0a5913898)

</details>

# Overall Task Expectations:
- The tasks should not be overly complex, but they do require a good understanding of federated learning, Docker, and the FL framework
- For both tasks, the deliverables should be clearly documented, and the code should be easy to run and well-commented

**Links:**
- [Flower AI](https://flower.ai/)
- [Breast Cancer Wisconsin (Diagnostic) - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## Dataset Attribution

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset  
**Source:** UCI Machine Learning Repository  
**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
**Full-Citation:** Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

