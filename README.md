[![DOI](https://zenodo.org/badge/596545346.svg)](https://zenodo.org/badge/latestdoi/596545346)

[![Documentation Status](https://readthedocs.org/projects/kgproject/badge/?version=latest)](https://kgproject.readthedocs.io/en/latest/?badge=latest)

# KGproject

KGProject is a knowledge graph about scientific papers, it’s been enriched with
information about authors obtained from Wikidata and about organizations
from ROR.

You can make queries to our KG to get the next information about the 30 papers:
- Title of the papers
- Authors of the papers
- Organizations acknowledged
- Information about the authors, such as:
   - The organization they're members of.
   - Their date of birth
   - Occupation
   - Country of citinzenship
   - Official website
   - Gender
   - Place of study
- Information about the organizations and publishers acknowledged, such as:
   - The location
   - Official website 
   - ROR URI 
- The probability of belonging to each topic.

*NOTE: Some information may not be found because it might not be available for the specific instance on the information sources used to enrich our KG.

## Requirements

- Docker
- Docker compose

## How to execute it
Our enriched knowledge graph is in a turtle file (.ttl), to get it and to be able to query it, you can download this repository and change directories until you are inside the 'compose' directory:

```
cd ./development/docker/compose
```
Then execute the next command:
```
docker-compose up --build
```
At the beginning of the execution of the prior command, you'll see something like this:
```
compose-jena-fuseki-1  | ###################################
compose-jena-fuseki-1  | Initializing Apache Jena Fuseki
compose-jena-fuseki-1  | 
compose-jena-fuseki-1  | Randomly generated admin password:
compose-jena-fuseki-1  | 
compose-jena-fuseki-1  | admin=zw2qhz63j5QfuIS
compose-jena-fuseki-1  | 
compose-jena-fuseki-1  | ###################################
```
Make sure you keep that password.

Then wait till you get this message "compose-rdfs-1 exited with code 0", so now you will have the .ttl file inside the directory ./compose/ttls

Then head over to localhost:3030 on your browser and introduce the password, then upload the .ttl file and now you can start to query it.

To stop the containers use:
```
docker-compose down
```

## Example queries

