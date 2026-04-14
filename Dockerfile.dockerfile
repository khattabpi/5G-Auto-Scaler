FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt  # ← added --no-cache-dir (keeps image smaller)

COPY . .

# Run data generation first, then train + serve
CMD ["python", "traffic_forecaster.py"]