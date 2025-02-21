# Use an official Python runtime as a base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port (update this if your app uses a different port)
EXPOSE 8000

# Run the application (modify based on your framework)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
