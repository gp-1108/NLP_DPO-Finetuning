# Step 1: Use Python base image
FROM python:3.9

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the project files into the working directory
COPY ./dataset_visualization /app/dataset_visualization
COPY ./dataset_generation /app/dataset_generation

# Step 5: Update the core link to dataset_generation/core folder
RUN rm -rf /app/dataset_visualization/core
RUN ln -s /app/dataset_generation/core /app/dataset_visualization/core

# Step 4: Install Python dependencies
RUN pip install -r /app/dataset_visualization/requirements.txt

# # Step 6: Expose the port Flask will run on
EXPOSE 5005

# # Step 8: Run the Flask app
CMD ["python3", "dataset_visualization/run.py", "/app/dataset_generation/data", "/app/dataset_generation/prompts/rules.txt"]