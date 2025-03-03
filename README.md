# KGRag


the main function of kgrag should have
1. insert: this function should insert a list of documents into knowledge base. 
More, the procedure should as following:
 - given a list of documents, check whether the current one was indexed
 - if not, check the document format
   - if it is json, read its items, construct 
     - document - entities
     - document - triples 
     - document - relations 
     - knowledge graph (entity should contains source document labels)
     - in order to deduplicate, should build a vector pair  entity - vector 

2. once have query, use llm to generate a logical chain, looks like e1- r1-e2-...-en
3. use chain to traverse kg, with level traverse
   looks like
    - Algorithm
    - Stack[logical-chain] S:
    - u = s.pop
    - candidate = []
    - if u.type is entity:
      - res = kg.find(u).similarity_top(k)
      - candidate.append(res)
    - if u.type is relation:
      - res = [e.relations for e in candidate].similarity_top(k)
      - candidate = []
      - candidate.append(res)
    if u is empty
      - return candidate
4. combine candidate and query to generate answer
