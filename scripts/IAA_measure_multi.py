import pandas as pd
from sklearn.metrics import cohen_kappa_score, multilabel_confusion_matrix, classification_report, precision_recall_fscore_support, confusion_matrix, plot_confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import krippendorff

"""
Calculate inter annotator agreement measures for three annotators
"""
#first download the input file from annotation sheet (google sheet) and turn into a .csv file
infile = pd.read_csv('../data_annotation/anno_ewt_wiki_annotation_3_annotators.csv')
anno1 = [str(i).lower() for i in infile['annotator1']]
anno2 = [str(i).lower() for i in infile['annotator2']]
anno3 = [str(i).lower() for i in infile['annotator3']]
outfile = open('../wiki+ewt_anno_master_results.txt', 'w')

#the category ? is a placeholder for un-annotated instances (those that are FL mistakes or don't fit the guidelines)
labels =['aspectual', 'causal',
         'degree', 'description',
        'comparison', 'possibility', 'negation', '?']
extended_labels = labels + ['descriptioncomparison',
                        'descriptiondegree', 'aspectualcausal', 'comparisondegree']

# cohen's kappa treating as exact match, no multilabel
ck_general = cohen_kappa_score(anno1, anno2, labels=extended_labels)
outfile.write(f"Cohen's Kappa 1-2: treating as exact match, no multi-label: {ck_general}\n")
outfile.write(f"Cohen's Kappa 1-3: treating as exact match, no multi-label: {cohen_kappa_score(anno1, anno3, labels=extended_labels)}\n")
outfile.write(f"Cohen's Kappa 2-3: treating as exact match, no multi-label: {cohen_kappa_score(anno2, anno3, labels=extended_labels)}\n")


print(f"Cohen's Kappa treating as exact match, no multi-label: {ck_general}\n")

#make into list of list, for multilabel data
anno1 = [[item.strip() for item in a.split('+')] for a in anno1]
anno2 = [[item.strip() for item in a.split('+')] for a in anno2]
anno3 = [[item.strip() for item in a.split('+')] for a in anno3]

print(anno1)
print(anno2)
print(anno3)
print(len(anno1))
same_subcat, overlapping_subcat = 0, 0
diff_subcats, one_na, both_na =  0, 0, 0
total = 0

# raw counts for categories
for i in range(len(anno1)):
    if set(anno2[i]) == set(anno3[i]) and len(set(anno2[i])) ==1 and len(set(anno3[i])) ==1 and anno2[i][0] == '?':
        both_na += 1
    elif set(anno2[i]) != set(anno3[i]) and (anno2[i][0] == '?' or anno3[i][0] == '?'):
        one_na += 1
    elif set(anno2[i]) == set(anno3[i]):
        same_subcat += 1
    elif len(set(anno2[i]) & set(anno3[i]))  ==0:
        diff_subcats += 1
    else:
        overlapping_subcat += 1

print(f'same subcat: {same_subcat}, diff_subcats: {diff_subcats}, overlapping subcat {overlapping_subcat}, one_na {one_na}, bothna {both_na}')

#calculate cohen's kappa on a per-category basis
to_dict = lambda x: {k: [1 if k in y else 0 for y in x] for k in labels}
#dict is with key=label_name, value=binary array of instances labeled with key
anno1_dict = to_dict(anno1)
anno2_dict = to_dict(anno2)
anno3_dict = to_dict(anno3)
print(anno1_dict)
cohen_dict = {k: cohen_kappa_score(anno1_dict[k], anno2_dict[k]) for k in labels}
cohen_avg = np.mean(list(cohen_dict.values()))

print(cohen_dict)
print(cohen_avg)
outfile.write(f"Cohen's Kappa for each category: {cohen_dict}\n")
outfile.write(f"Cohen's Kappa averaged across categories: {cohen_avg} \n")

# krippendorff
k_alpha_dict = {k: krippendorff.alpha([anno1_dict[k], anno2_dict[k], anno3_dict[k]]) for k in labels}
print(k_alpha_dict)
k_alpha_avg = np.mean(list(k_alpha_dict.values()))
print(f'k_alpha_avg: {k_alpha_avg}\n')

#calculate partially correct via accuracy, treating annotation2 as the gold standard
#binarize the examples
outfile.write("\n--------partial correct examples via accuracy, treating #2 as the gold----")
anno1_binary=[[0] * len(labels) for _ in range(len(anno1))]
anno2_binary=[[0] * len(labels) for _ in range(len(anno2))]

anno1_index = [[labels.index(a) if a in labels else -1 for a in anno] for anno in anno1]
anno2_index = [[labels.index(a) if a in labels else -1 for a in anno] for anno in anno2]

for i, ex in enumerate(anno1_index):
    for ind in ex:
        anno1_binary[i][ind] = 1
print('ANNO1 BINARY', anno1_binary)
for i, ex in enumerate(anno2_index):
    for ind in ex:
        anno2_binary[i][ind] = 1
print(anno1_index)
print(anno2_index)

#use formula:
# accuracy = 1/n*SUM(labels predicted correctly / gold union labels predicted)
avg_accuracy = 1 / len(anno1_index)
avg_prec = 1 / len(anno1_index)
avg_recall = 1 / len(anno1_index)

sum = 0
prec_sum = 0
recall_sum = 0
for i, example in enumerate(anno1_index):
    corrects = 0
    for ind in example:
        if ind in anno2_index[i]:
            corrects+=1
    sum += (corrects/len(set(example+anno2_index[i])))
    prec_sum += (corrects/len(set(anno2_index[i])))
    recall_sum += (corrects / len(set(example)))
    # print("sum +=", (corrects/len(set(example+anno2_index[i]))))
avg_accuracy *= sum
avg_prec *= prec_sum
avg_recall *= recall_sum
outfile.write(f'\nPartial: avg_accuracy={avg_accuracy} \navg_prec=, {avg_prec}\navg_recall= {avg_recall}')

#exact match calculation no frequency
corrects = 0
for i , example in enumerate(anno1_index):
    if set(example) == set(anno2_index[i]):
        corrects +=1
exact_match = corrects/len(anno1_index)
outfile.write(f'\nexact match= {exact_match}')


# multilabel confusion matrix
mcm = multilabel_confusion_matrix(anno2_binary, anno1_binary)
outfile.write(f"\n\nMultilabel Confusion Matrix, Label-based:\nLabels:  {labels} \nFor coordinate representation please see README \n  {mcm}")

true_negative = mcm[:, 0,0]
true_positive = mcm[:,1,1]
false_negative = mcm[:, 1, 0]
false_positive = mcm[:, 0,1]
outfile.write(f"\nBy label:\nmcm accuracy -{(true_positive + true_negative) /(true_positive + true_negative + false_negative+false_positive)}")
outfile.write(f"\nmcm precision - {(true_positive) /(true_positive + false_positive)}")
outfile.write(f"\nmcm recall -  {(true_positive) /(true_positive + false_negative)}")


#standard confusion matrix, treating multiple label as one label
anno2_index_flat = [''.join(s) for s in anno2]
anno1_index_flat = [''.join(s) for s in anno1]
print('confusion matrix input example', anno1_index_flat)
normal_mcm = confusion_matrix(anno2_index_flat, anno1_index_flat, labels=extended_labels)
outfile.write(f"\n\nConfusion Matrix (by exact match), also see image output\n  {normal_mcm}")

#image output
plt.figure(figsize = (15,15))
sns.heatmap(normal_mcm, annot=True,  xticklabels=extended_labels, yticklabels=extended_labels)
# plt.show()
plt.savefig("wiki+ewt_anno_result.jpg")

clf_report = classification_report(np.array(anno2_binary), np.array(anno1_binary), target_names=labels)
outfile.write(f"\n\nClassification report\n{clf_report}")

outfile.close()