FROM python:3.11.5
# USER root
WORKDIR /chainlit-app

COPY requirements.txt .
RUN pip install -r ./requirements.txt
COPY . .

CMD ["chainlit", "run", "chatbot.py", "--host", "0.0.0.0", "--port", "8000"]
