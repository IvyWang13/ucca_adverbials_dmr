USES OF SCRIPTS IN THIS FOLDER

##IAA_measure.py
- is used to calculate inter-annotator agreements measures, in order: Cohen's Kappa, Partial accuracy, Exact match, and confusion matrix.
- the input should be two lists/columns from .csv file, of the two annotators
- The inputs should be preprocessed such that the category terms are standardized, i.e. Comparison and not Comparative, etc.

## mrp_filter_wsj.py
- takes the ucca.mrp from the MRP 2020 data and filter out those that come from WSJ corpus.
- from the resulting data, sample 100 items randomly.
- outputs mrp_ucca_no_wsj.csv which includes these 100 sentences.

## mrp_process.py
- can be used on any .mrp files of ucca format, is used to extract the D/Adverbial instances from each annotated sentence
- outputs three .csv columns: sentence-id, D-item, and the original sentence.
- The D-items, if multi-word, may not be in order.
- When input comes from mrp_ucca_no_wsj.mrp (the randomly sampled 100 sentences), output is stored in the file mrp_ucca.csv


##folder ewt_wiki_100/
- includes the more human readable full annotation of each sentence in mrp_ucca.csv results.
- Annotators can refer to the respective graphs for full foundational layer decisions made previously.