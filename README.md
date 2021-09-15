## Data used for pilot annotation

### folder ewt_wiki_100/
- includes data in the the more human-readable full annotation format of each sentence in mrp_ucca.csv results. This is the first round of annotation.
- Annotators can refer to the respective graphs for full foundational layer decisions made previously.

### Folder ewt_wiki_2_100/
- includes data the second set of annotation.

# USES OF SCRIPTS IN THIS FOLDER

## IAA_measure.py
- Used to calculate inter-annotator agreements measures, in order: Cohen's Kappa, Partial accuracy, Exact match, and confusion matrix.
- The input should contain at least two lists/columns from .csv file, of the two annotators
- The inputs should be preprocessed such that the category terms are standardized, i.e. Comparison and not Comparative, etc.
- The outputs for WSJ annotations can be found in WSJ_MRP_UCCA_IAA.txt

## mrp_filter_wsj.py
- takes the ucca.mrp from the MRP 2020 data and filter out those that come from WSJ corpus.
- from the resulting data, sample 100 items randomly.
- outputs mrp_ucca_no_wsj.csv which includes these 100 sentences.

## mrp_process.py
- can be used on any .mrp files of ucca format, is used to extract the D/Adverbial instances from each annotated sentence
- outputs three .csv columns: sentence-id, D-item, and the original sentence.
- The D-items, if multi-word, may not be in order.
- When input comes from mrp_ucca_no_wsj.mrp (the randomly sampled 100 sentences), output is stored in the file mrp_ucca.csv

## Reading multilabel confusion matrix
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
- In multilabel confusion matrix , the count of true negatives is [0,0], false negatives is [1,0], true positives is [1,1] and false positives is [0,1].

## Reading Exact match confusion matrix
- see e.g. wiki+ewt_anno_result.jpg
