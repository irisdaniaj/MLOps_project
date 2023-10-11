# Use the official Python image as the base image
FROM python:3.11-slim

# Copy the requirements file into the container
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
COPY src /src

# Install the dependencies
RUN pip install --no-cache-dir -r /requirements.txt
