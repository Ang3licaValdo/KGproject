[![DOI](https://zenodo.org/badge/596545346.svg)](https://zenodo.org/badge/latestdoi/596545346)
# KGproject

KGProject is a knowledge graph about scientific papers, information such
as title, author, organizations acknowledged and projects mentioned on
the papers is the foundation of this graph, it’s also been enriched with
information about authors obtained from Wikidata and about organizations
from ROR.

You can make queries to our KG to get the next information about the 30 papers:
- Title of the papers
- Authors of the papers
- Organizations acknowledged
- Information about the authors such as:
   - The organization they're members of.
   - Their date of birth
   - Occupation
   - Country of citinzenship
   - Official website
   - Gender
   - Place of study
- Information about the organizations and publishers acknowledged such as:
   - The location
   - Official website 
   - ROR URI 
- The probability of belonging to each topic.

## How to get the .ttl file
Our enriched knowlegded is in a turtle file (.ttl), to get it, you can download this repository and change directories until you are inside the 'compose' directory:

```
cd ./development/docker/compose

```
Then execute the next command:
```
docker-compose up --build

```
