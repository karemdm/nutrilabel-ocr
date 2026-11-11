# nutrilabel-ocr container image
#
# Build:
#   docker build -t nutrilabel .
#
# Run the web interface (then open http://localhost:7860):
#   docker run --rm -p 7860:7860 nutrilabel
#
# Use the CLI on a local photo or folder (arguments override the default CMD):
#   docker run --rm -v $PWD:/data nutrilabel nutrilabel read /data/photos --csv /data/results.csv

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-spa \
        tesseract-ocr-ita tesseract-ocr-nld tesseract-ocr-por \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir ".[tesseract,rapidocr]"

EXPOSE 7860

CMD ["nutrilabel", "ui", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
