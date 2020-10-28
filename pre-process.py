import pandas as pd
ham = pd.read_csv("ham.txt",delimiter='\t+',header=None) 
ham.insert(0, 'label', '0')
spam = pd.read_csv("spam.txt",delimiter='\t+',header=None) 
spam.insert(0, 'label', '1')
