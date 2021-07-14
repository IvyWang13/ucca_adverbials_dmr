import jsonlines

mrp_path = '/Users/ivywang/Downloads/ucca_adverbials/'
input_file = 'mrp_ucca_no_wsj.mrp'
out_file = open('mrp_ucca_no_wsj.csv', 'w')
out_file.write(f'sent_id\tsentence\td-unit\n')

#process jsonline file format
total = 0
with jsonlines.open(mrp_path+input_file) as reader:

    for obj in reader:

        print(obj['id'] , ' ')

        sentence = obj['input']
        # print(obj['nodes'])
        # print(obj['edges'])
        for edge in obj['edges']:
            if edge['label'] == 'D':
                # print(edge)
                source_node =obj['nodes'][edge['source']]
                target_node = obj['nodes'][edge['target']]
                d_unit = ""
                if 'anchors' not in target_node:
                    while 'anchors' not in target_node:
                        #search for the edge whose source is the current target node
                        targets = []
                        for e in obj['edges']:
                            if e['source'] == target_node['id']:
                                targets.append(obj['nodes'][e['target']])
                        #have list of targets
                        # print(targets)
                        for t in targets:
                            if 'anchors' in t:
                                # print(sentence[t['anchors'][0]['from'] :t['anchors'][0]['to']])
                                d_unit += sentence[t['anchors'][0]['from'] :t['anchors'][0]['to']]
                                d_unit += ' '
                                target_node = t
                            else:
                                target_node = t
                                break
                else:
                    for anchor in target_node['anchors']:
                        # print(sentence[anchor['from'] : anchor['to']])
                        d_unit += sentence[anchor['from'] : anchor['to']]
                # print(d_unit)
                # out_file.write(obj['id'])
                # out_file.write("\t")
                # out_file.write(sentence)
                # out_file.write("\t")
                # out_file.write(d_unit)
                # out_file.write('\n')



        total+=1


print("total:", total)

