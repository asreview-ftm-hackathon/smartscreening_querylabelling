# smartscreening_querylabelling

[Follow The Money](https://www.ftm.nl/) (FTM) is a Dutch platform for investigative journalism. 
Specifically, they investigate the money trails of "people, systems, and organisations that engage in (financial/economic) misconduct". 
Journalists at FTM are [currently attempting](https://www.ftm.nl/dossier/shell-papers) to use Freedom of Information (WOB) requests to obtain all of the communication between the Dutch government and the oil giant Shell in the period 2005-2019. 
This is expected to amount to roughly 150.000 documents. As is getting more common in Open-source investigations, the sheer volume of data that needs to be analyzed poses a significant challenge for investigators.

FTM and ASReview organised this Hackathon to see if the Open-source Active Learning system [ASReview](https://www.nature.com/articles/s42256-020-00287-7) can help cut down on the time needed to find relevant documents.
This software is currently used by scientists to find relevant research papers for writing systematic reviews. First, the scientist collates a large amount of possibly relevant research papers as input to the software. Then the software iteratively presents the scientist with these papers to label as (ir)relevant, while it learns to predict this label for the remaining papers.
This allows the scientist to find relevant papers a lot faster than going through the collection by themselves.

Translating the use of this software to investigative journalism is not straightforward. 
One of the complicating differences between the use cases, is that the journalist did not collate the collection of documents on possible relevance prior to using the software. 
On the contrary, the data is often a sparsely relevant mess of unknown and dissimilar documents. This means that the active learning system will require much more labelling of data before it has learnt the features which predict relevance.

> In order for ASReview to adequately address some of these problems, **we believe that there need to be more options for oracle-knowledge injection before- and during the screening process.**

One possibility is to allow the user to label subsets of the data based on a particular search query. 
Maybe the user knows that any document that contains 'John' in the title, or 'environmental pollution' in the body is relevant.
In that case, having functionality within ASReview to label all documents which match a query like (title:John OR body:environmental pollution) as 'relevant' can speed up the screening process significantly.

> **As a first step towards this kind of functionality, we developed a script which updates the labels of a tabular collection of documents (.csv/.xlsx) based on a search query combining boolean operations.**


We aimed at keeping the code as general as possible, to make copying the functionality for any other projects easy.

## Getting started


> The script should work with these dependencies:
  * Python (version 3.9.2)
  * *numpy* (version 1.20.2)
  * *pandas* (version 1.3.3)
  * *spacy* (version 3.2.0)
    * 'en_core_web_sm' language model (installed with `python -m spacy download en_core_web_sm`)
  * *scikit-learn* (version 1.0.1)
  * *pythonds* (version 1.2.1)

>Make sure you have the following in a single folder:
  * The `querylabelling.py` **script**.
  * A **dataset** containing a collection of textdocuments in .csv or .xlsx format.
    * If using .csv, make sure it is ;-delimited and utf-8 encoded.
    * Each row represents a document.
    * Each column contains a separately searchable section of the document.
      * The first row of the dataset contains the names of the columns.
    * If the data is already labelled, the column should be named `'label_included'` and loaded.
      * Labels used are: -1 = **unlabelled**, 0 = **'exclude'**, 1 = **'include'**.
      
>Navigate to the folder in your CLI, then run the script with `python querylabelling.py`.  

> Enter the name of your dataset. (e.g. `dataset.csv`)  

> Name the columns you want to load. (e.g `title,body,label_included`)

> Enter the label to apply to the queried subset. (`include`,`exclude`,`unlabelled`)

> Enter the search query. (e.g `(title:John OR body:environmental pollution)`)
* The formatting for the query is strict:
  * The boolean operations allowed are `AND`,`OR` and `NOT`.
  * Each operation needs to match enclosing parentheses:
    * `(__ AND __)`
    * `(__ OR __)`
    * `(NOT __)`
  * Search terms can contain a specification for the column to search in (e.g `title:John`)
    * The column name is to the left of the first `:`, the search term is to the right.
    * If no column is specified, the search term can be in any column.
  * Search terms are case-insensitive.
  * Search terms consisting of multiple words specify an exact sequence.
  * Search terms are stripped of all non-alphanumeric characters, and 'lemmatized'. This means that a search term like `'looking after'` will match `'look after'` and `'looked after'` within the data. Lemmatization is done with an English language model. For other languages look [here](https://spacy.io/usage/models).
  

>**Examples**  
> Documents with `Y` in the abstract, but without `Z` in any column: `(abstract:Y AND (NOT Z))`  
> Documents excluding `A` and `B C`: `((NOT A) AND (NOT B C))`  
> Documents with`D` or `E` in the title: `(title:D OR title:E)`


## Authors 
Written by Matthew van der Meer and Evi Hendrikx for the Follow The Money ASReview Hackathon 2021.


## License
The content in this repository is published under the MIT license.

## Contact
For any questions or remarks, please send an email to e.h.h.hendrikx@uu.nl or matthew.vdmeer@gmail.com 
