# Use a Python base image with Jupyter installed
FROM python:3.9

# Set working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY groundwater_imputation/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter Notebook
RUN pip install jupyter

# Copy the local files into the container
COPY . /app

# Expose the Jupyter Notebook port
EXPOSE 8888

# Command to run Jupyter Notebook and mount the local directories
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.allow_origin='*'", "--NotebookApp.notebook_dir=/app/groundwater_imputation/src/imputation_api"]
