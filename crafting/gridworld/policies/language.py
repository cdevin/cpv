import pdb
import numpy as np

GRAMMAR = {
    '<SENT>': ('*', ['ChopTreePolicy', 'ChopRockPolicy', 'EatBreadPolicy', 'BuildHousePolicy', 'MakeBreadPolicy']),
    'ChopTreePolicy': ('chop_tree(<DET>, <COLOR>, <LOC>)', ["<PICKUP_AXE> chop <DET> <COLOR> tree <LOC>"]),
    'ChopRockPolicy': ('chop_rock(<DET>, <COLOR>, <LOC>)', ["break <DET> <COLOR> rock <LOC>"]),
    'EatBreadPolicy': ('eat_bread(<DET>, <COLOR>, <LOC>)', ["eat <DET> <COLOR> bread <LOC>"]),
    'BuildHousePolicy': ('build_house(<DET>, <COLOR>, <LOC>)', ["build <DET> <COLOR> house <LOC>"]),
    'MakeBreadPolicy': ('make_bread(<DET>, <COLOR>, <LOC>)', ["make <DET> <COLOR> bread <LOC>"]),
    '<PICKUP_AXE>': ('', ['', 'pickup an axe and ']),
    '<COLOR>': ('<COLOR>', ['', 'red', 'blue', 'green', 'yellow', 'black', 'white']),
    '<LOC>': ('<LOC>', ['', 'on your left', 'on your right', 'behind you', 'in front of you']),
    '<DET>': ('<DET>', ['a', 'two'])
}

ONTOLOGY = {
    'red': 'red',
    'blue': 'blue',
    'green': 'green',
    'yellow': 'yellow',
    'black': 'black',
    'white': 'white',
    'on your left': 'left',
    'on your right': 'right', 
    'behind you': 'behind',
    'in front of you': 'front', 
    'a': '1', 
    'two': '2',
    '': '*' 
}

def expand(non_terminal): 

    # Get logical form and expansion options for non-terminal. 
    logical_form, options = GRAMMAR[non_terminal]

    # Randomly select top-level expansion. 
    template = np.random.choice(options)

    # If leaf, replace logical form with item from ontology. 
    if logical_form in {'<COLOR>', '<LOC>', '<DET>'}:
        logical_form = ':'.join([logical_form, ONTOLOGY[template]])

    # Will hold expanded sentence and logical forms. 
    sent = []
    logical_forms = [logical_form]

    # Expand until no non-terminals remain. 
    for word in template.split():

        # Non-terminal. 
        if word in GRAMMAR: 
            word_expansion, logical_expansion = expand(word)

            sent.append(word_expansion)
            logical_forms.extend(logical_expansion)
        else: 
            sent.append(word)

    sent = ' '.join(sent)

    return (sent, logical_forms)

def collapse(logical_forms): 

    assert(logical_forms[0] == '*')
    logical_form = logical_forms[1]

    for i in range(2, len(logical_forms)): 
       
        # Consume empty forms. 
        if logical_forms[i] == '':
            pass

        else: 
            
            # Ground variable. 
            form_type, value = logical_forms[i].split(':')
            logical_form = logical_form.replace(form_type, value)

    return logical_form

def simple_language(policy_list):

    for i in range(30): 

        # Sample sentence and logical form from grammar. 
        sentence, logical_forms = expand('<SENT>')   
        sentence = ' '.join(sentence.split())

        # Resolve logical form. 
        logical_form = collapse(logical_forms)

        print([sentence, logical_form])

    pdb.set_trace()

    return sentence
