import jsonlines
import random

"""
Filter the WSJ data out of the MRP data and randomly select 100 sentences from the remainder
"""
#change this path accordingly
mrp_path = '/mrp/2020/cf/training/'
input_file = 'ucca.mrp'
out_file = open('../data_annotation/mrp_ucca_no_wsj_2.mrp', 'w')
total = 0
writer = jsonlines.Writer(out_file)

#dedup from the first 100 randomly selected sentences
compare_file = '../data_annotation/mrp_ucca_no_wsj.mrp'
existing_ids = []
with jsonlines.open(compare_file) as reader:
    for obj in reader:
        existing_ids.append(obj['id'])

json_list=[]
with jsonlines.open(mrp_path+input_file) as reader:

    for obj in reader:

        if obj['source'] == 'wsj' or obj['id'] in existing_ids:
            continue
        json_list.append(obj)
        total += 1
print("total = ", total)
json100 = random.sample(json_list, 100)
print(json100, len(json100))


writer.write_all(json100)