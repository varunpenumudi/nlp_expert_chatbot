import time
import pickle
import pandas as pd
import yake
from nltk.corpus import stopwords


# Read DATASET
df = pd.read_feather(R'..\filtered_data\filtered_data.feather')


# ----------------------------------
# EXTRACT KEYWORDS
# ----------------------------------
y = yake.KeywordExtractor(stopwords=stopwords.words('english'))

abstr_kws = ['']*df['abstract'].count()

start = time.time()
for i, abstr in enumerate(df['abstract']):
    if i%1000 == 0:
        print(f"{i} abstracts processed")
    
    keywords = [kw for kw,score in y.extract_keywords(abstr)]
    abstr_kws[i] = ', '.join(keywords)

print(f"Took total of {time.time() - start} Seconds for extraction")



# ----------------------------------
# SAVE KEYWORDS (PICKLE)
# ----------------------------------
with open('abstract_keywords.pickle', 'wb') as file:
    pickle.dump(abstr_kws, file)


# ----------------------------------
# SAVE KEYWORDS (Pandas DataFrame)
# ----------------------------------
with open('abstract_keywords.pickle', 'rb') as file:
    keywords = pickle.load(file)

df['keywords'] = keywords # add keywords column
df.to_feather('filtered_with_keywords.feather')