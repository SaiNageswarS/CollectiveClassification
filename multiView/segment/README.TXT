-- Multi-View Synthetic Datasets

We provide here a set of synthetic multi-view text datasets, constructed from the single-view BBC and BBCSport news article corpora by splitting articles into related segments of text.  

From the two news corpora we constructed 6 new datasets containing 2-4 views as follows:
   1. We split each raw document into segments. This was done by separating the documents into paragraphs, and merging sequences of consecutive paragraphs until 1-4 segments of text remained, such that each segment was at least 200 characters long. Each segment is logically associated with the original document from which it was obtained.
   2. The segments for each document were randomly assigned to views, with the restriction that at most one segment from each document was assigned to the same view. 


This archive contains 6 different datasets, organised by the originating corpus. Each dataset contains 2-4 views, as indicates by the file prefix (e.g. bbc_seg1of2.*,  bbc_seg2of2.*). The view data files have the following formats:
    - *.mtx: Term frequencies stored in a sparse term-document matrix in Matrix Market format.
    - *.terms: List of content-bearing terms in the corpus, with each line corresponding to a row of the corresponding term-document matrix.
    - *.docs: List of article identifiers, with each line corresponding to a column of the corresponding term-document matrix. Note that the article identifiers correspond across views.

In addition annotated labels are provided for the news articles. The identifiers in these files correspond to the article identifiers above:
    - bbc.clist: Non-overlapping (single label) annotated topic classes, for datasets originating from the BBC corpus.
    - bbcsport.clist: Non-overlapping (single label) annotated topic classes, for datasets originating from the BBCSport corpus.    
   

These datasets are made available for non-commercial and research purposes only, and all data is provided in pre-processed matrix format. Note that stemming (Porter algorithm), stop-word removal (stop word list) and low term frequency filtering (count < 3) have already been applied to the data. 

All rights, including copyright, in the content of the original abstracts are owned by the original authors.  

For further details please contact derek.greene@ucd.ie

http://mlg.ucd.ie/datasets/segment.html


