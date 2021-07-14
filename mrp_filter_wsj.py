import jsonlines
import random

mrp_path = '/Users/ivywang/Downloads/ucca_adverbials/mrp/2020/cf/training/'
input_file = 'ucca.mrp'
out_file = open('mrp_ucca_no_wsj.mrp', 'w')
total = 0
writer = jsonlines.Writer(out_file)
json_list=[]
with jsonlines.open(mrp_path+input_file) as reader:

    for obj in reader:

        if obj['source'] == 'wsj':
            continue
        json_list.append(obj)
        total += 1
print("total = ", total)
json100 = random.sample(json_list, 100)
print(json100, len(json100))


writer.write_all(json100)