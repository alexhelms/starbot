FROM python:3.9.6-buster

ARG DISCORD_TOKEN
ENV DISCORD_TOKEN $DISCORD_TOKEN

RUN pip install -r requirements.txt

COPY starbot /app/

CMD ["python", "/app/starbot/main.py"]