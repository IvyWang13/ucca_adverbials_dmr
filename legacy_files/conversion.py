from ucca import convert
import re
from ucca import constructions, core, visualization
import matplotlib
import pandas as pd
matplotlib.use("Agg")

import matplotlib.pyplot as plt
def is_composite_unit(text):
    return '[' in text and ']' in text

def is_simple_adverb(text):
    print('adverb: ', text)
    if len(text.split()) ==1:
        return re.search('ly$', text) is not None
    return 0

def is_negation(text):
    print(f'negation, {text}')
    return re.search(r"n't|not|no", text.lower()) is not None

def is_modal_aux(text):
    text = text.lower()
    if text == 'could' or text == 'would' or text == 'might':
        return 1
    return 0

def get_word_count(text):
    words = text.split()
    spare_list = ['[',']','E','C','R']
    words = [w for w in words if w not in spare_list]

    print(f'word list: {words}')
    return len(words)

def strip_nested_brackets(sent):
    sent = sent.split()
    new_list = []
    # insert space before ] and after [
    for i, item in enumerate(sent):
        if '[' in item:
            new_list.extend(['[',item[1:]])
        elif ']' in item:
            new_list.extend([item[:-1], ']'])
        else:
            new_list.extend(item)

    accepted = ['[',']','C','E','D','A','P','S','H','G','N','F','R','U', ' ', 'A*','D*', 'T','Q', 'L', 'P*', 'S*'] # ucca annotation labels
    new_list = [item for item in new_list if len(item) < 3 and item in accepted]

    return ' '.join(new_list)

def get_top_level(text):
    text = text.split()
    ind = 0
    new_str = ''
    while ind  < len(text)-1:
        if text[ind] == '[' and text[ind+1] in ['A','S','D','T','P']:
            new_str += text[ind]
            new_str += text[ind+1]
            ind += 2
            left = 1
            while left != 0:
                if text[ind] == '[':
                    left += 1
                elif text[ind] == ']':
                    left -= 1
                ind += 1
            new_str += ']'
        else:
            new_str+= text[ind]
            ind+= 1
    print(new_str)
    return new_str

# out_file = open('D_categories.csv', 'a')
# config_out_file = open('D_configurations.csv', 'a')
# out_file.write(f'passge_id\tnode_id\tnumChildren\tedge_tag\tedge_id\tedge\tcomposite\tsimple_adverb\tnegation\tmodal_aux\tword_count\tadjective\tdiscontinuous\n')
infile = pd.read_csv('../data_annotation/Copy_UCCA-adverbials - D_categories_en.csv')
# node_id_list = [str(i) for i in infile['node_id']]
# node_id_list = [i[:5] if len(i)>5 else i for i in node_id_list]
# print(node_id_list)
# infile['node_id'] = node_id_list
plain_text_col = open('../plain_text_column.csv', 'w')
totald_in_psg = 0
for i in range(1, 3, 1):
    # change the following to the desired passage folder
    converted_psg = convert.file2passage(f'./UCCA_English-WSJ-master/xml/wsj_0001.{i}.xml')
    print(f'passage number {i}')
    print(converted_psg) # passage object converted from the xml format
    layers_list = list(converted_psg.layers)
    layer0 = layers_list[0]
    layer1 = layers_list[1]

    for j, one in enumerate(layer1.all): # iterate through the node
        # get the edge label of this node

        for e in one.incoming:
            if e.tag =='D' : # or e.tag == "E":
                print(f'number {j}, \tID {one.ID},\t num_children {len(list(one.children))}')
                totald_in_psg += 1
                d_text = str(e.child)

                # print(f'tag : {e.tag}, "  : ", id {e.ID}, text {d_text}')
                # out_file.write(f'{i}\t{one.ID}\t{len(list(one.children))} \t{e.tag}\t{e.ID}\t{e.child}\t')

                # lexically categorize
                # out_file.write(f'{int(is_composite_unit(d_text))}\t')
                # out_file.write(f'{int(is_simple_adverb(d_text))}\t')
                # out_file.write(f'{int(is_negation(d_text))}\t')
                # out_file.write(f'{int(is_modal_aux(d_text))}\t')
                # out_file.write(f'{get_word_count(d_text)}\t')
                # out_file.write('\t\n')

                # configurationally categorize
                # config_out_file.write(f'{i}\t{one.ID}\t{len(list(one.children))} \t{e.tag}\t{e.ID}\t{e.child}\t')
                for parent in one.parents:
                    # parent: the node which has incoming edges to one
                    # having multiple parents means it appears in multiple constructions --> remote edges
                    # one node could have multiple parents. count these different parents, but one parent-oneedge will only be counted once
                    for pe in parent.incoming:
                        # config_out_file.write(f'{parent.ID}\t{parent.tag}\t')
                        # config_out_file.write(f'{pe.ID}\t{pe.child}\t{strip_nested_brackets(str(pe.child))}\t{get_top_level(strip_nested_brackets(str(pe.child)))}\n')
                        print(f'parent: {parent.ID}\t{parent.tag}\t')
                        print(f'parent edge id: {pe.ID}\n{pe.child}')
                        print(f'try convert to plain text : {(pe.child.to_text())}')
                        plain_text_col.write(f'{pe.child.to_text()}\n')

                        break

    #visual
    # f = plt.figure(figsize=(60,14))
    # visualization.draw(converted_psg, node_ids=True)
    # f.savefig("graph.png")
# print(f'the avg D items per passage is {totald_in_psg/(942-911)}') # 4.111 fr
# for english avg 3.889 for 36-62
# for english 3.909 for 286- 318
# 814-846 english avg D 3.970
#  880-909 english avg D : 7.067
#  968-998 english avg D : 6.258
#
# English side:
#
# Chapter1: Passages 36-62
# Chapter2: Passages 286-318
# Chapter3: Passages 814-846
# Chapter4: Passages 880-909
# Chapter5: Passages 968-998
# French side:
#
# Chapter1: Passages 77-103 : 4.111
# Chapter2: Passages 416-448: 3.030
# Chapter3: Passages 764-796: 2.788
# Chapter4: Passages 848-877: 5.033
# Chapter5: Passages 911-941: 5.161



