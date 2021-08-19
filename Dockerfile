FROM python:3.9.6-buster

ARG DISCORD_TOKEN
ENV DISCORD_TOKEN $DISCORD_TOKEN

COPY starbot /app/
WORKDIR /app/starbot
RUN pip install -r requirements.txt

CMD ["python", "/app/starbot/main.py"]