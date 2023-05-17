# Usa una imagen oficial de Apache Spark como base
FROM apache/spark:latest

# Instala las dependencias de Python necesarias
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install findspark requests beautifulsoup4 && \
    pip3 install -U scikit-learn

RUN pip install --no-cache-dir findspark pyspark rdflib lxml beautifulsoup4 fuzzywuzzy torch torchvision torchaudio transformers nltk SPARQLWrapper

WORKDIR /rdf_dir

# Copia la aplicación de Python al contenedor
COPY . .

# Establece el punto de entrada del contenedor para ejecutar la aplicación de Python
ENTRYPOINT ["spark-submit", "--master", "local[*]", "stableversionroberta.py"]
