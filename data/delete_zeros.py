for directory in ["MQ2008", "MQ2007"]:
    for fold in ["Fold"+str(i) for i in range(1,6)]:
        for f in ["train.txt","test.txt","vali.txt"]:
            path = directory+"/"+fold+"/"+f
            queries = []
            qid = ""
            for line in open(path):
                s = line.split()
                # Check if new query, and if so, append a new list of documents
                if s[1] != qid:
                    queries.append([])
                    qid = s[1]
                # Append document to current query
                queries[-1].append(s)
            of = open(path,"w")
            for query in queries:
                # Check if there are non-zero documents in query
                non_zero = False
                for doc in query:
                    if doc[0]!="0":
                        non_zero = True
                        break
                # If there are, write query into file
                if non_zero:
                    for doc in query:
                        of.write(doc[0])
                        for v in doc[1:]:
                            of.write(" "+v)
                        of.write("\n")


