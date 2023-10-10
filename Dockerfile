# Use the official Python image as the base image
FROM python:3.11-slim

# Copy the requirements file into the container
COPY requirements.txt /requirements.txt
COPY mlops/setup.py /mlops/setup.py
COPY mlops/src /mlops/src

# Install the dependencies
RUN pip install --no-cache-dir -r /requirements.txt