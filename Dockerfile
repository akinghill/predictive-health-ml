FROM python:3.8-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./models/heart_disease.pkl /deploy/
WORKDIR /deploy/
RUN pip3 install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]