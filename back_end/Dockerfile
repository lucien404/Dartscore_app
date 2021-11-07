FROM python:3.9
MAINTAINER luning.li
COPY . /app
WORKDIR /app
RUN rm -rf venv
RUN apt update
RUN apt install libgl1-mesa-glx -y
RUN pip install flask
RUN pip install opencv-contrib-python
RUN pip install matplotlib
RUN pip install torch
RUN pip install torchvision
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]