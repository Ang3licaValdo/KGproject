
"""# Librerias generales

Instalación de spark
"""


"""Librerias complementarias de spark"""

import os


import findspark
findspark.init()
findspark.find()

import glob
import requests

from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Learning_Spark") \
    .getOrCreate()

"""Librerias para generar RDFs y manipular XMLs"""

from bs4 import BeautifulSoup

"""# GROBID

Se envia el PDF a GROBID para obtener un TEI XML
"""

def grobid_request(file):
  f = {'input': open(file, 'rb')}
  try:
    response = requests.post('https://kermitt2-grobid.hf.space/api/processFulltextDocument', files=f)
    if response.status_code == 200:
      return response.content 
  except requests.exceptions.RequestException:
    print("ALERTA: Grobid no logró procesar el archivo")

  return None

"""# Extraction

Crea la clase Paper para contener la información obtenida de cada uno de los PDFs
"""

class Paper:
  def __init__(self, title, authors, organizations, abstract, publishers, acknowledgement, topic_distribution, acknowledgementPeople, acknowledgementOrganizations, projects):
    # Initialize the attributes with the provided values
    self.title = title
    self.authors = authors
    self.organizations = organizations
    self.abstract = abstract
    self.publishers = publishers
    self.acknowledgement = acknowledgement
    self.topic_distribution = topic_distribution 
    self.acknowledgementPeople = acknowledgementPeople
    self.acknowledgementOrganizations = acknowledgementOrganizations
    self.projects = projects

  def set_authors(self, authors):
    self.authors = authors
    return self 

  def set_topic_distribution(self, topic_distribution):
    self.topic_distribution = topic_distribution
    return self 

  def set_attributes_from_acknowledgement(self, roberta_results):
    self.acknowledgementPeople = roberta_results.people
    self.acknowledgementOrganizations = roberta_results.organizations
    self.projects = roberta_results.projects
    return self 

  def set_organizations(self, organizations):
    self.organizations = organizations
    return self 

  def set_publishers(self, publishers):
    self.publishers = publishers
    return self

class Author:
  def __init__(self, full_name, institution):
    self.full_name = full_name
    self.institution = institution 
  def set_institution(self, institution):
    self.institution = institution
    return self

def process_author(author_element):
    name_element = author_element.find("persName")
    if not name_element:
        return None

    first_name_element = name_element.find("forename", {"type": "first"})
    surname_element = name_element.find("surname")
    full_name = f"{getattr(first_name_element, 'text', '')} {getattr(surname_element, 'text', '')}" \
        if first_name_element and surname_element else ""

    affiliation_element = author_element.find("affiliation")
    if not affiliation_element:
        return Author(full_name, "")

    institution_element = affiliation_element.find("orgName", {"type": "institution"})
    if institution_element:
        institution = institution_element.text
    else:
        raw_affiliation_element = affiliation_element.find("note", {"type": "raw_affiliation"})
        institution = raw_affiliation_element.text if raw_affiliation_element else ""

    return Author(full_name, institution)

def process_affiliation(affiliation_element):
    org_name_element = affiliation_element.find("orgName", {"type": "institution"})
    if org_name_element is not None:
        institution = org_name_element.text
    else:
        raw_affiliation_element = affiliation_element.find("note", {"type": "raw_affiliation"})
        if raw_affiliation_element is not None:
            institution = raw_affiliation_element.text
        else:
            return ""

    return institution

import random
def process_xml_file(xml_file):
  soup = BeautifulSoup(xml_file, "xml")

  title = soup.title.text if soup.title else "paper"+str(random.randint(1000, 9999))
  authors = [process_author(author) for author in soup.find_all("author") if process_author(author)]
  organizations = list(set([process_affiliation(affiliation) for affiliation in soup.find_all("affiliation") if affiliation.text and process_affiliation(affiliation)]))
  abstract = soup.abstract.text if soup.abstract else ""
  publisher = list(set([publisher.text for publisher in soup.find_all("publisher") if publisher.text]))
  acknowledgement = soup.find('div', {'type': 'acknowledgement'}).text if soup.find('div', {'type': 'acknowledgement'}) else ""
  
  return Paper(title, authors, organizations, abstract, publisher, acknowledgement, "", "", "", "")

"""# Topic Model

Obtenemos los distintos topics a partir de los abstracts de los papers, usanod topiz modeling LDA
"""

#Library
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Global variable used by both functions
count_vectorizer = CountVectorizer()

import nltk
from nltk.corpus import stopwords

# Descargar las stopwords si no se han descargado previamente
nltk.download('stopwords')

# Obtener la lista de stopwords en el idioma deseado
stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
    # Dividir el texto en palabras
    words = text.split()
    # Eliminar las stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Unir las palabras filtradas en un nuevo texto
    filtered_text = ' '.join(filtered_words)
    return filtered_text

# Creates the topics
def topic_creator(abstracts_list):
  
  X = count_vectorizer.fit_transform(abstracts_list)
  # Creating 3 topics
  lda = LatentDirichletAllocation(n_components=12, random_state=0)
  lda.fit(X)

  feature_names = count_vectorizer.get_feature_names_out() 
  # for topic_id, topic in enumerate(lda.components_):
  #     print(f"Topic {topic_id}:")
  #     print(" ".join([feature_names[i] for i in topic.argsort()[:-6:-1]]))
  
  return lda

# Given an abstract it returns the similarity of that abstract to each topic
def topic_distribution(paper, lda):
  
  new_doc_bow = count_vectorizer.transform([remove_stop_words(paper.abstract)])

  topic_distribution = lda.transform(new_doc_bow)

  # Dictionary containing topics and the distribution of the abstract with each of them
  distribution_of_topics = {} 

  for topic_idx, topic_prob in enumerate(topic_distribution[0]):
    distribution_of_topics[topic_idx] = topic_prob
  

  return distribution_of_topics

"""# ROR request"""


from fuzzywuzzy import fuzz
import requests

# class Organization:
#   def __init__(self, ror_id, name, homepage, location):
#     self.ror_id = ror_id
#     self.name = name
#     self.homepage = homepage
#     self.location = location
class Organization:
    def __init__(self, ror_id, name, homepage, location):
        self.ror_id = ror_id
        self.name = name
        self.homepage = homepage
        self.location = location

    def __getstate__(self):
        # Devolver un diccionario con los atributos necesarios para reconstruir el objeto
        return {
            'ror_id': self.ror_id,
            'name': self.name,
            'homepage': self.homepage,
            'location': self.location
        }

    def __setstate__(self, state):
        # Restaurar los valores de los atributos a partir del diccionario
        self.ror_id = state['ror_id']
        self.name = state['name']
        self.homepage = state['homepage']
        self.location = state['location']

def get_info_orgs(orgs_names):
  orgs = []
  for org_name in orgs_names:
      org = ROR_request(org_name)  
      orgs.append(org)  
  return orgs

def get_info_authors_org(authors):
    return [author.set_institution(ROR_request(author.institution)) for author in authors if author.institution]

def ROR_request(org_name):
    api_key = '015w2mp89'
    url = 'https://api.ror.org/organizations'
    headers = {'Authorization': f'Token {api_key}'}
    params = {'query': org_name}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['number_of_results'] > 0:
          organization_name = data['items'][0]['name']
          similarity_score = fuzz.ratio(org_name.lower(), organization_name.lower())
          print(similarity_score)
          if similarity_score > 45:
            
            ror_id = data['items'][0]['id']

            if 'addresses' in data['items'][0] and len(data['items'][0]['addresses']) > 0:
                location = data['items'][0]['addresses'][0]['city']
            else:
                location = None

            if 'links' in data['items'][0] and len(data['items'][0]['links']) > 0:
                homepage = data['items'][0]['links'][0]
            else:
                homepage = None

            return Organization(ror_id, organization_name, homepage, location)
        else:
            print(f"No results found for {org_name}")
            return Organization(None, org_name, None, None)
    else:
        print(f'Request failed with status code {response.status_code}')
    return Organization(None, org_name, None, None)


# orga = ROR_request("Springer")
# print(orga.name)

"""# Roberta Model

Instalando librerías para Roberta
"""


from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re #for regular expressions
import nltk

nltk.download('stopwords')
nltk.download('words')

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.corpus import words

"""Objeto Roberta donde se contienen las organizaciones y las personas encontradas en los acknowledgments"""

# class Roberta_results:
#     def __init__(self, organizations, people, projects):
#         self.organizations = organizations
#         self.people = people
#         self.projects = projects
#     def set_organizations(self, organizations):
#       self.organizations = organizations
#       return self
class Roberta_results:
    def __init__(self, organizations, people, projects):
        self.organizations = organizations
        self.people = people
        self.projects = projects

    def set_organizations(self, organizations):
        self.organizations = organizations
        return self

    def __getstate__(self):
        # Devolver un diccionario con los atributos necesarios para reconstruir el objeto
        return {
            'organizations': self.organizations,
            'people': self.people,
            'projects': self.projects
        }

    def __setstate__(self, state):
        # Restaurar los valores de los atributos a partir del diccionario
        self.organizations = state['organizations']
        self.people = state['people']
        self.projects = state['projects']

"""Función para obtener las organizaciones y personas mencionadas en los acknowledgments"""

def roberta(acknowledgments):
    list_of_organizations = []
    list_of_people = []
    miscelaneous = []

    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")

    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    result = nlp(acknowledgments)
    

    for resulting in result:
        if resulting['entity_group'] == 'PER':
            list_of_people.append(resulting['word'])
            #print(resulting['word'])
        elif resulting['entity_group'] == 'ORG':
            list_of_organizations.append(resulting['word'])
            #print(resulting['word'])
        elif resulting['entity_group'] == 'MISC':
            miscelaneous.append(resulting['word'])

    list_of_projects = []
    words_without = []
    new_acknowledgments = acknowledgments
    
    #removes people
    for person in list_of_people:
        
        person = str(person)
        if person in new_acknowledgments:
            new_acknowledgments = new_acknowledgments.replace(person, "")


    #removes organizations
    for organization in list_of_organizations:
        organization = str(organization)
        if organization in new_acknowledgments:
            new_acknowledgments = new_acknowledgments.replace(organization, "")


    #tokenizing the acknowlegdments
    words_with = new_acknowledgments.split()

    #removing unwanted characters
    par_to_remove = "(" 
    words_without = [word.replace(par_to_remove, "") for word in words_with]
    

    char_to_remove = ")" 
    words_without2 = [word.replace(char_to_remove, "") for word in words_without]
    

    dot_to_remove = "."
    words_without3 = [word.replace(dot_to_remove, "") for word in words_without2]
    

    apos_to_remove = "'"
    words_without4 = [word.replace(apos_to_remove, "") for word in words_without3]
    

    coma_to_remove = ","
    words_without5 = [word.replace(coma_to_remove, "") for word in words_without4]
    
    
    #Define the regular expression pattern
    pattern = r'^[a-zA-Z0-9_/_-]+$'
    pattern_misc_minor = "^[a-zA-Z_]+$"
    english_words = set(words.words())
    stop_words = set(stopwords.words('english'))

    words_without6 = [word for word in words_without5 if not re.match(pattern_misc_minor,word)]
    

    for word in words_without6:

        match = re.match(pattern, word)

        # Check if the string matches the pattern
        if match:

            if word.lower() not in english_words and word.lower() not in stop_words:
              list_of_projects.append(word)

    return Roberta_results(get_info_orgs(list_of_organizations),list_of_people,list_of_projects)

"""# Wikidata request"""


from rdflib import Graph, Literal, URIRef, BNode
from SPARQLWrapper import SPARQLWrapper2, TURTLE


def getURIFromString(label: str, graph: Graph) -> URIRef:
    # Load the Wikidata RDF data into the graph
    endpoint = 'https://query.wikidata.org/sparql'
    # Query for the resource URI based on the given name and type
    query = """
        SELECT ?s
        WHERE {
            ?s rdfs:label "%s"@en ;
               wdt:P31 <http://www.wikidata.org/entity/Q5> .
        }
        LIMIT 1
    """ % (label)
    # Execute the SPARQL query
    sparql = SPARQLWrapper2(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(TURTLE)
    results = sparql.queryAndConvert()
    # Return the first URI found, if any
    if (results.bindings == []):
        return None
    subject = results.bindings[0]["s"].value
    expandFromURI(subject, graph)
    return subject


def expandFromURI(s: str, graph: Graph):
    # Load the Wikidata RDF data into the graph
    endpoint = 'https://query.wikidata.org/sparql'
    # Query for the resource URI based on the given name and type
    query = """
        PREFIX p: <http://www.wikidata.org/prop/direct/>
        SELECT ?s ?p ?o
        WHERE {
            <%s> ?p ?o .
            FILTER (
                ?p = <http://www.wikidata.org/prop/direct/P21>
                || ?p = <http://www.wikidata.org/prop/direct/P463>
                || ?p = <http://www.wikidata.org/prop/direct/P27>
                || ?p = <http://www.wikidata.org/prop/direct/P569>
                || ?p = <http://www.wikidata.org/prop/direct/P106>
                || ?p = <http://www.wikidata.org/prop/direct/P856>
                || ?p = <http://www.wikidata.org/prop/direct/P19>
                || ?p = <http://www.wikidata.org/prop/direct/P69>
            )
        }
    """ % (s)
    # Execute the SPARQL query
    sparql = SPARQLWrapper2(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(TURTLE)
    results = sparql.queryAndConvert()
    # Bind Namespaces to RDFLib Graph
    s = URIRef(s)
    for triple in results.bindings:
        p = URIRef(triple["p"].value)
        o = triple["o"]
        if (o.type == "uri"):
            o = URIRef(o.value)
        elif (o.type == "literal"):
            o = Literal(o.value, lang=o.lang, datatype=o.datatype)
        graph.add((s, p, o))


# print(getURIFromString("Albert Einstein", "http://www.wikidata.org/entity/Q5"))

graph = Graph()
if getURIFromString("Jofadsfsa", graph):
  print("hola")
print(graph.serialize(format='turtle'))

"""# RDF generator"""

import time
from rdflib import Graph, Namespace, Literal, RDF, URIRef, BNode
from rdflib.namespace import FOAF, DCTERMS, RDFS
import random
import string
import uuid



def create_knowledge_graph(papers, topics, similar_to_dict):
    # Crea un nuevo grafo RDF
    graph = Graph()
    # Define el espacio de nombres "project"
    project = Namespace("http://www.project.com/")
    graph.bind("project", project)
    wd = Namespace("http://www.wikidata.org/entity/")
    graph.bind("wd", wd)
    wdt = Namespace("http://www.wikidata.org/prop/direct/")
    graph.bind("wdt", wdt)
    schema = Namespace("https://schema.org/")
    graph.bind("schema", schema)
    owl = Namespace("http://www.w3.org/2002/07/owl#")
    graph.bind("owl", owl)

    graph.parse("./ontology.ttl", format="turtle") 
    
    graph.add((project.term(""), RDF.type, owl.Ontology))

    

    
    for topic_id, topic in enumerate(topics.components_):
        topic_uri = project.term(generate_identifier("topic"+str(topic_id)))
        graph.add((topic_uri, RDF.type, owl.NamedIndividual))
        graph.add((topic_uri, wdt.P31, wd.Q200801)) # wd:topic
        graph.add((topic_uri, RDFS.label, Literal(f"Topic {topic_id}")))
        for i in topic.argsort()[:-6:-1]:
          graph.add((topic_uri, schema.keywords, Literal(feature_names[i])))

    for paper in papers:
        # Crea un URI único para el documento
        paper_uri = project.term(generate_identifier(paper.title))
        graph.add((paper_uri, RDF.type, owl.NamedIndividual))
        print(paper.title)
        # Agrega triples al grafo para cada atributo del Paper
        graph.add((paper_uri, wdt.P31, wd.Q13442814)) # wd:scholary article
        graph.add((paper_uri, RDFS.label, Literal(paper.title))) # wd:title
        for author in paper.authors:
            author_uri = getURIFromString(author.full_name, graph)
            if author_uri:
              author_uri = URIRef(author_uri)
            else:
              author_uri = project.term(generate_identifier(author.full_name))
              graph.add((author_uri, RDF.type, owl.NamedIndividual))
            graph.add((author_uri, wdt.P31, wd.Q482980)) # wd:author
            graph.add((author_uri, RDFS.label, Literal(author.full_name))) # wd:name
            graph.add((author_uri, wdt.P7137, paper_uri)) # wd:acknowledged
            #----Agregar Intitution
            if author.institution:
              organization_uri = project.term(generate_identifier(author.institution.name))
              graph.add((organization_uri, RDF.type, owl.NamedIndividual))
              graph.add((organization_uri, wdt.P31, wd.Q43229)) # wd:organization
              graph.add((organization_uri, RDFS.label, Literal(author.institution.name))) # wd:name
              if author.institution.ror_id:
                  graph.add((organization_uri, wdt.P6782, URIRef(author.institution.ror_id))) # wd:ror_id
              if author.institution.homepage:
                  graph.add((organization_uri, wdt.P856, URIRef(author.institution.homepage))) # wd:official website
              if author.institution.location:
                  graph.add((organization_uri, wdt.P159, Literal(author.institution.location))) # wd:headquarters location
              graph.add((author_uri, wdt.P463, organization_uri))

        for organization in paper.organizations:
            organization_uri = project.term(generate_identifier(organization.name))
            graph.add((organization_uri, RDF.type, owl.NamedIndividual))
            graph.add((organization_uri, wdt.P31, wd.Q43229)) # wd:organization
            graph.add((organization_uri, RDFS.label, Literal(organization.name))) # wd:name
            if organization.ror_id:
                  graph.add((organization_uri, wdt.P6782, URIRef(organization.ror_id))) # wd:ror_id
            if organization.homepage:
                graph.add((organization_uri, wdt.P856, URIRef(organization.homepage))) # wd:official website
            if organization.location:
                graph.add((organization_uri, wdt.P159, Literal(organization.location))) # wd:headquarters location
            graph.add((organization_uri, wdt.P7137, paper_uri)) # wd:acknowledged

        for publisher in paper.publishers:
            publisher_uri = project.term(generate_identifier(publisher.name))
            graph.add((publisher_uri, RDF.type, owl.NamedIndividual))
            graph.add((publisher_uri, wdt.P31, wd.Q43229)) # wd:organization
            graph.add((publisher_uri, RDFS.label, Literal(publisher.name))) # wd:name
            if publisher.ror_id:
                  graph.add((publisher_uri, wdt.P6782, URIRef(publisher.ror_id))) # wd:ror_id
            if publisher.homepage:
                graph.add((publisher_uri, wdt.P856, URIRef(publisher.homepage))) # wd:official website
            if publisher.location:
                graph.add((publisher_uri, wdt.P159, Literal(publisher.location))) # wd:headquarters location
            graph.add((paper_uri, wdt.P123, publisher_uri))

        for person in paper.acknowledgementPeople:
            person = person.strip('\n').replace('\n', ' ')
            person_uri = getURIFromString(person, graph)
            if person_uri:
              person_uri = URIRef(person_uri)
            else:
              person_uri = project.term(generate_identifier(person))
              graph.add((person_uri, RDF.type, owl.NamedIndividual))
            graph.add((person_uri, wdt.P31, wd.Q482980)) # wd:author
            graph.add((person_uri, RDFS.label, Literal(person))) # wd:name
            graph.add((person_uri, wdt.P7137, paper_uri)) # wd:acknowledged

        for organization in paper.acknowledgementOrganizations:
            organization_uri = project.term(generate_identifier(organization.name))
            graph.add((organization_uri, RDF.type, owl.NamedIndividual))
            graph.add((organization_uri, wdt.P31, wd.Q43229)) # wd:organization
            graph.add((organization_uri, RDFS.label, Literal(organization.name))) # wd:name
            if organization.ror_id:
                  graph.add((organization_uri, wdt.P6782, URIRef(organization.ror_id))) # wd:ror_id
            if organization.homepage:
                graph.add((organization_uri, wdt.P856, URIRef(organization.homepage))) # wd:official website
            if organization.location:
                graph.add((organization_uri, wdt.P159, Literal(organization.location))) # wd:headquarters location
            graph.add((organization_uri, wdt.P7137, paper_uri)) # wd:acknowledged
        for topic, prob in paper.topic_distribution.items():
            topic_uri = project.term(generate_identifier("topic"+str(topic)))
            blnk = BNode(generate_identifier(paper.title+"topic"+str(topic)))
            graph.add((paper_uri, wdt.P921, blnk))
            graph.add((blnk, project.value, topic_uri))
            graph.add((blnk, project.probability, Literal(prob)))

        for paper_project in paper.projects:
            graph.add((paper_uri, wdt.P355, Literal(paper_project)))
        time.sleep(1)
    
    # Asigna las relaciones (similar)
    for key, URIs in similar_to_dict.items():
      for URIsuj in URIs:
        for URIobj in URIs:
          if URIsuj != URIobj:
            graph.add((project.term(URIsuj), wdt.P1659, project.term(URIobj))) #wd.related work

    return graph
          


import uuid
import hashlib

def generate_identifier(texto):
    # Generar el hash MD5 del texto
    hash_md5 = hashlib.md5(texto.encode()).hexdigest()

    # Crear un UUID a partir del hash MD5 y darle formato con guiones bajos
    uuid_generado = uuid.UUID(hash_md5).hex

    # Insertar guiones bajos en la posición adecuada
    id = '_'.join([uuid_generado[:8], uuid_generado[8:12], uuid_generado[12:16], uuid_generado[16:20], uuid_generado[20:]])

    return id

"""# Spark"""

# Crear sesión de Spark
# spark = SparkSession.builder.appName("rdf-to-ttl").getOrCreate()

# Se obtienen PDFs del directorio
#pdf_files = glob.glob("./drive/Shareddrives/Ciencia Abierta/Papers/*.pdf")
# pdf_files = glob.glob("./*.pdf")
pdf_files = glob.glob("./PDFs/*.pdf")
print(pdf_files)

# Crear RDD de archivos PDF
pdf_files_rdd = spark.sparkContext.parallelize(pdf_files)
# print("pdf_files_rdd: ", pdf_files_rdd.count())

# Aplicar función a cada elemento del RDD y recopilar XMLs TEI en un nuevo RDD
xml_files_rdd = pdf_files_rdd.map(grobid_request)
# print("xml_files_rdd: ", xml_files_rdd.count())

# Aplicar función a cada elemento del RDD y recopilar XMLs TEI en un nuevo RDD
papers_rdd = xml_files_rdd.map(process_xml_file)
# print("papers_rdd: ", papers_rdd.count())

# Se obtiene una lista de todos los abstracts
abstracts_list = papers_rdd.map(lambda paper: remove_stop_words(paper.abstract)).collect()

# La lista se pasa a la funcion topic_creator para generar los temas
topics = topic_creator(abstracts_list)

# Imprime temas
feature_names = count_vectorizer.get_feature_names_out() 
for topic_id, topic in enumerate(topics.components_):
  print(f"Topic {topic_id}:")
  print(" ".join([feature_names[i] for i in topic.argsort()[:-6:-1]]))

topic_distribution_rdd = papers_rdd.map(lambda paper: topic_distribution(paper, topics))
# roberta_results_rdd = papers_rdd.map(lambda paper: roberta(paper.acknowledgement))
# roberta_results_orgs_rdd = roberta_results_rdd.map(lambda r_results: r_results.set_organizations(get_info_orgs(r_results.organizations)))
orgs_rdd = papers_rdd.map(lambda paper: get_info_orgs(paper.organizations))
publishers_rdd = papers_rdd.map(lambda paper: get_info_orgs(paper.publishers))
authors_with_institution_rdd = papers_rdd.map(lambda paper: get_info_authors_org(paper.authors))


t_d = topic_distribution_rdd.collect()
# r_r = roberta_results_rdd.collect()
orgs = orgs_rdd.collect()
pubs = publishers_rdd.collect()
awi = authors_with_institution_rdd.collect()

papers = papers_rdd.collect()

for paper, t, o, p, a in zip(papers, t_d, orgs, pubs, awi):
    paper.set_topic_distribution(t)
    paper.set_attributes_from_acknowledgement(roberta(paper.acknowledgement))
    paper.set_organizations(o)
    paper.set_publishers(p)
    paper.set_authors(a)

def similar_to(papers):
  similar_to_dict = {}

  for paper in papers:
    max_key = None
    max_probability = float('-inf')

    for key, topic_prob in paper.topic_distribution.items():
      if topic_prob > max_probability:
        max_key = key
        max_probability = topic_prob

    if max_key not in similar_to_dict:
      similar_to_dict[max_key] = []

    similar_to_dict[max_key].append(generate_identifier(paper.title))

  return similar_to_dict

for paper in papers:
    print("-"*200)
    print("Title:", paper.title)
    print("Authors:")
    for author in paper.authors:
        print("-", author.full_name)
        if(author.institution):
          print(" -",author.institution.name)
    print("Organizations:")
    for organization in paper.organizations:
        print("-", organization.name)
        if organization.ror_id:
          print("  -", organization.ror_id)
    print("Abstract:", paper.abstract)
    print("Publisher:")
    for publisher in paper.publishers:
        print("-", publisher.name)
        if publisher.ror_id:
          print("  -", publisher.ror_id)
    print("Acknowledgement:", paper.acknowledgement)
    print("Acknowledgement People:", paper.acknowledgementPeople)
    print("Acknowledgement Organizations:")
    for organization in paper.acknowledgementOrganizations:
        print("-", organization.name)
        if organization.ror_id:
          print("  -", organization.ror_id)
    print("Project IDs:", paper.projects)
    print("Topic Distribution:")
    print(paper.topic_distribution)
    print("-"*200)

"""# New Section"""

similar_to_dict = similar_to(papers)
grafo = create_knowledge_graph(papers, topics, similar_to_dict)
turtle_data = grafo.serialize(format="turtle")
file_path = "archives/mi_archivo.ttl"

# Guardar el grafo RDF en un archivo
with open(file_path, "w") as file:
    file.write(turtle_data)

