FROM nia50-1

WORKDIR /app

# .: outside of the container(current), ./: inside the container(/app)
COPY . ./

EXPOSE 80

RUN pip install -r requirements.txt

# CMD ["python", "app.py"]