# syntax=docker/dockerfile:1

FROM python:3.11

WORKDIR /code

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y libgl1-mesa-dev && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 50505
RUN cd $APP_PATH

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
