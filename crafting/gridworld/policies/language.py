import pdb
import numpy as np

# TODO: Start from grammar with top level SENT, then make sure to return logical representation of goal spec.
# TODO Remake simple_language
GRAMMAR = {
    '<SENT>': ['ChopTreePolicy', 'ChopRockPolicy'],
    'ChopTreePolicy': ["<PICKUP_AXE> chop <DET> <COLOR> tree <LOC>"],
    'ChopRockPolicy': ["break <DET> <COLOR> rock <LOC>"],
    'EatBreadPolicy': ["eat <DET> <COLOR> bread <LOC>"],
    'BuildHousePolicy': ["build <DET> <COLOR> house <LOC>"],
    'MakeBreadPolicy': ["make <DET> <COLOR> bread <LOC>"],
    'PickAndPlace': ["", '],
    '<PICKUP_AXE>': ['', 'pickup a <COLOR> axe and '],
    '<COLOR>': ['', 'red', 'blue', 'green', 'yellow', 'black', 'white'],
    '<LOC>': ['', 'on your left', 'on your right', 'behind you', 'in front of you'],
    '<DET>': ['a', 'two']
}

def expand(non_terminal): 

    # Randomly select top-level expansion. 
    template = np.random.choice(GRAMMAR[non_terminal])
    sent = []

    # Expand until no non-terminals remain. 
    for word in template.split():

        # Non-terminal. 
        if word in GRAMMAR: 
            sent.append(expand(word).strip())
        else: 
            sent.append(word)

    return ' '.join(sent)

def simple_language(policy_list):

    new_sentence = []
    for pol in policy_list:

        # Add expanded form. 
        new_sentence.append(expand(pol).strip())
    
    sentence = ' then '.join(new_sentence)
    
    print(sentence)
    pdb.set_trace()

    return sentence
