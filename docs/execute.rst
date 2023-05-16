How to execute it
=================

Our enriched knowledge graph is in a turtle file (.ttl), to get it and
to be able to query it, you can download this repository and change
directories until you are inside the ‘compose’ directory:

::

   cd ./end point/docker/compose

Then execute the next command:

::

   docker-compose up --build

At the beginning of the execution of the prior command, you’ll see
something like this:

::

   compose-jena-fuseki-1  | ###################################
   compose-jena-fuseki-1  | Initializing Apache Jena Fuseki
   compose-jena-fuseki-1  | 
   compose-jena-fuseki-1  | Randomly generated admin password:
   compose-jena-fuseki-1  | 
   compose-jena-fuseki-1  | admin=zw2qhz63j5QfuIS
   compose-jena-fuseki-1  | 
   compose-jena-fuseki-1  | ###################################

Make sure you keep that password.

Then wait till you get this message “compose-rdfs-1 exited with code 0”,
so now you will have the .ttl file inside the directory ./compose/ttls

Then head over to localhost:3030 on your browser and introduce the
password, then upload the .ttl file and now you can start to query it.

To stop the containers use:

::

   docker-compose down
