import parserFile


if __name__ == "__main__":
    # Open the first Reuters data set and create the parser
    filename = "dataset/reut2-000.sgm"
    parser = parserFile.ReutersParser()

    # Parse the document and force all generated docs into
    # a list so that it can be printed out to the console
    doc = parser.parse(open(filename, 'rb'))
    print list(doc)[0][0]
    #pprint.pprint(list(doc))