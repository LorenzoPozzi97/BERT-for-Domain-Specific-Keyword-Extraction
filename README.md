# BERT-for-Domain-specific-Keyword-Extraction
In this project the BERT model was used for the task of extracting keywords in a domain-specific scenario. More specifically, a custom dataset will be used to identify statistical terms in a corpus generated from 8 academic textbooks.

## Dataset Specifics
- 8 textbooks on Statistics
- ~800 different keywords
- ~24k paragraphs
- ~1M words

The dataset has 8 columns, each row contains a single paragraph extracted from the book:
- **Relevance**, if at least least a keyword is present in the paragraph is '1', '0' otherwise.
- **Tags**, the keywords found in the paragraph.
- **Heading**, the chapter title of the paragraph.
- **Seg**, the unique identifies of the chapter title.
- **Sentence**, the paragraph.
- **Enc Tags**, binary represenation of the paragraph, non-keywords are marked as '0', while keywords are '1'.
- **Enc Heading**, BERT embeddings of the chapter title.
- **Enc Sentence**, BERT embeddings of the paragraph.

Snippet of the Dataset:

<img src="https://github.com/LorenzoPozzi97/BERT-for-Domain-Specific-Keyword-Extraction/blob/8382a41dba8989787c1d61ad26ad5c817267fc0c/Dataset%20Snippet.png" alt="drawing" width="500"/>

## Keyword Extraction Model
The model used consist of:
- BERT encoder
- Bidirectional LSTM
- Linear Classifier

<img src="https://github.com/LorenzoPozzi97/BERT-for-Domain-Specific-Keyword-Extraction/blob/537533512f55b73aa9203380cd1fac66bb8e6835/model.png" alt="drawing" width="600"/>
