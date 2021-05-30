FROM python:3.6-buster
WORKDIR /app
RUN pip install cmake
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install numpy==1.19.5
COPY . .
CMD ["python", "main.py"]