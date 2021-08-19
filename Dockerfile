FROM python:3.9.6-buster

ARG DISCORD_TOKEN
ENV DISCORD_TOKEN=$DISCORD_TOKEN

RUN apt-get update && \
    apt-get install -y git

RUN git clone https://github.com/alexhelms/starbot.git

WORKDIR /starbot
RUN pip install -r requirements.txt

CMD ["python", "starbot/main.py"]