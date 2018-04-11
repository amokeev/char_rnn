import config

def generate_vocabulary():
    input_file = config.get("input_file")
    data = open(input_file, 'r').read()
    vocabulary = sorted(list(set(data)))
    print("Got vocabulary of leght %d"%len(vocabulary))
    f = open(config.get('vocabulary_file'), 'w')
    for v in vocabulary:
        if v == "\n":
            v="<BREAK>"
        if v != "":
            f.write("%s\n" % v)
    f.close()

def load_vocabulary():
    vlines = open(config.get('vocabulary_file'), 'r').read().splitlines()
    vocabulary = []
    for i in range(len(vlines)):
        if vlines[i] == "<BREAK>":
            vlines[i] = ('\n')
        vocabulary.append(vlines[i])
    return vocabulary