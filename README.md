We show how substituting letters with their corresponding confusable tricks spam filters into classifying spam emails as ham emails. 

The data was used for this experiment below:
- spam-classification: data downloaded from [spam-classification](https://github.com/subhashbylaiah/spam-classification) on November 24, 2019.
- enron-spam: data downloaded from [enron-spam](http://www2.aueb.gr/users/ion/data/enron-spam/) on November 24, 2019.

It can be executed with following commands, as indicated in experiment.py:
- ./experiment.py spam_ham_dataset.csv A
- ./experiment.py spam_ham_dataset.csv B
- ./experiment.py spam_ham_dataset.csv C

We also had a pre-process step to convert data from the “raw” format to the one used by the Python script to suit this setup by pooling the entire set of email messages in a single comma-separated values file such that each row is a full representation of an individual email’s information consisting of a “spam” or “ham” label, comma-separated from the email text. The reformatted file was parsed and converted to a pandas dataframe with no headers. The resulting dataframe is a simple framework that consists of two columns only: the email instance and the corresponding label, with the label appearing before the email instance. Each email message is a single space-delimited stream of words.
