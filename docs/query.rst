Example queries
===============
Query 1
--------

Get labels of articles and topics with their probabilities.

.. code-block:: sparql

   PREFIX : <http://www.project.com/>
   PREFIX wdt: <http://www.wikidata.org/prop/direct/>
   PREFIX wd: <http://www.wikidata.org/entity/>
   PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   SELECT ?articleLabel ?topicLabel ?probability
   WHERE {
     ?article wdt:P921 [
       :value ?topic ;
       :probability ?probability ;
     ] .
     ?topic rdfs:label ?topicLabel .
     ?article rdfs:label ?articleLabel .
   }
   LIMIT 1


Query 2
--------

Get labels of articles and topics with their probabilities and sort by probability.

.. code-block:: sparql

   PREFIX : <http://www.project.com/>
   PREFIX wdt: <http://www.wikidata.org/prop/direct/>
   PREFIX wd: <http://www.wikidata.org/entity/>
   PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   SELECT ?articleLabel ?topicLabel ?probability
   WHERE {
     ?article wdt:P921 [
       :value ?topic ;
       :probability ?probability ;
     ] .
     ?topic rdfs:label ?topicLabel .
     ?article rdfs:label ?articleLabel .
   }
   ORDER BY DESC(?probability)


Query 3
--------

Given the label of an article, get the related articles.

.. code-block:: sparql

   PREFIX : <http://www.project.com/>
   PREFIX wdt: <http://www.wikidata.org/prop/direct/>
   PREFIX wd: <http://www.wikidata.org/entity/>
   PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   SELECT ?label ?relatedlabel
   WHERE {
     ?article rdfs:label "Optimizing SPARQL Queries over Decentralized Knowledge Graphs" .
     ?article rdfs:label ?label .
     ?article wdt:P1659 ?related .
     ?related rdfs:label ?relatedlabel
   }
