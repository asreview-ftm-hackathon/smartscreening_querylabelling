# smartscreening_FUNnelyourdata_TheFUNnel


## The FUNnel
In this repository, we developed a function that can screen tabular data based on user's inputs. This input can have Boolean values about specified columns of the data.
For example: ("Follow AND ((NOT title:The) OR abstract:money))")

## The motivation
We developed this in assignment for the Hackaton ASReview for Follow the Money. It is hard for the current ASReview algorithm to deal with the dataset of Follow the Money. This is because it consists of a HUGE amount of documents, that are additionally noisy and possibly contain plenty irrelevant interactions, which is different from its usual input: already specifically selected datasets.
Therefore, we decided to create a search engine-like function that the user can use to select specific files from this (or any other tabular) dataset.
We aimed on making the code blocks as generalizable as possible, so they can easily be implemented in other projects/ datasets
## Getting started
Clone the repository, navigate to the folder in the CMD, and run main
You will be prompted to input the file path to the dataset and the search query.
The search query can contain ( __ AND __ ), ( __ OR __ ), and (NOT __ ), which can specify the columns for which a selection should be made (default: over all columns)
For example, if you want
* Only papers with x in the title:		
  - title:x
* Only papers with y in the abstract, but that do not contain z anywhere
  - (abstract:y AND (NOT z))
* Only papers that do not contain a and b
  - ((NOT a) AND (NOT b))

## Authors 
Written by Matthew van der Meer and Evi Hendrikx as an assignment in the Hackaton ASReview for Follow the Money.

## License
MIT

## Contact
For any questions or remarks, please send an email to e.h.h.hendrikx@uu.nl or matthew.vdmeer@gmail.com 
