FROM python:3.9.6-buster

ARG DISCORD_TOKEN
ENV DISCORD_TOKEN=$DISCORD_TOKEN

WORKDIR /app/starbot
COPY . .
RUN pip install -r requirements.txt

CMD ["python", "starbot/main.py"]