FROM nvcr.io/nvidia/pytorch:25.01-py3

# Set up working directory
WORKDIR /workspace

# Install any additional packages you might need
# Uncomment and modify as needed
# RUN pip install -U numpy pandas scikit-learn matplotlib

# Install all the requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set default command to bash
ENTRYPOINT ["/bin/bash"]
