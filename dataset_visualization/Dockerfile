# Step 1: Use Python base image
FROM python:3.9

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Clone the repo
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/gp-1108/NLP_DPO-Finetuning.git

# Step 4: Go to the dataset_visualization directory
WORKDIR /app/NLP_DPO-Finetuning/dataset_visualization

# Step 5: Update the core link to dataset_generation/core folder
RUN rm -rf /app/NLP_DPO-Finetuning/dataset_visualization/core
RUN ln -s /app/NLP_DPO-Finetuning/dataset_generation/core /app/NLP_DPO-Finetuning/dataset_visualization/core

# Step 4: Install Python dependencies
RUN pip install -r requirements.txt

# Step 6: Expose the port Flask will run on
EXPOSE 5001

# Step 8: Run the Flask app
CMD ["python3", "run.py", "/app/NLP_DPO-Finetuning/dataset_generation/data"]