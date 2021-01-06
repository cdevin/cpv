
POLICY2SENTENCE= {
    'ChopTreePolicy': "chop a tree",
    'ChopRockPolicy': "break a rock",
    'EatBreadPolicy': "eat a bread",
    'BuildHousePolicy': "build a house",
    'MakeBreadPolicy': "make bread",
}

def simple_language(policy_list):
    new_sentence = []
    for pol in policy_list:
        new_sentence.append(POLICY2SENTENCE[pol])
    sentence = ' then '.join(new_sentence)
    return sentence