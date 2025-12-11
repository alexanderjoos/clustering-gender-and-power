# clustering-gender-and-power
## How to set things up
- First of all, sorry if things are a bit messy! I did this all in colab cause I bought some sweet GPU time
- I have secrets in my notebook for my huggingface and namesor API keys, so go to the secrets bar in the left and set them or just hardcode in your own notebook
- I did some wonky stuff for the riveter download check out their repo link: https://github.com/maartensap/riveter-nlp

## The thick of it
## Your info
* NetID(s): aaj46
* Name(s): Alexander Joos
---

## Introduction and Hypothesis

####Introduction

I am currently in a communication and technology class, INFO 2450, this semester, and I wanted to center my project around communication dynamics since we have talked about a lot about technology’s afforcance of mediating and curating speech. Specifically, I wanted to analyze text detailing these furnished communications within networks. Much of the text that I interact with is in the form of shorter form messages through social media and email. I feel that these types of corpus are a step away from the more traditional literary analysis that we have looked at in class, as each text is part of a larger story, with both a sender and receiver. The idea of presenting your message to a specific recipient often changes the verbs and tones used to communicate messages, arrange meetings, or generally how one dresses up the content.

Social media is often more of a one-to-many type of communication, where people morph both image and textual content into an ideal self. This appeal can be seen as one that any person could look at, given the size of social networks. However, email narrows down this focus. Senders modify taglines, talk about events in certain ways, and manipulate verb choice based on the specific person that they are emailing, and thus use different patterns to communicate with different people.

To be able to extract ‘culture’ from a corpus like this, I gravitated towards the Enron emails dataset. Bounding the network to a workplace creates more measurable power markings through a corporate hierarchy, and the mix between business and personal communications binds the variety of a sender's outreach to the company cultural sphere. Implied receiver labels like work friends, work spouses, and ingratiators certainly exist in how each person conveys the message body. One can look at communication profiles of individuals talking up or down, as well as how a position may entail a certain communication persona.

Beyond coincidental and hierarchical factors, power expression also occurs through gender. Specifically, Prabhakaran et al. (2014) discuss a “gender environment,” defined as the gender makeup of message recipients, and illustrate its effect in determining power within communications. Steffensmeier et al. (2013) further explore female involvement in workplace crime partially within Enron, underscoring how gender interacts with corporate dynamics and conspiracy.

####Hypothesis

Building on the idea that communication in workplace networks reflects power dynamics, social roles, and identity curation, I hypothesize the following:

> "Individuals within a workplace network adjust their communication style, specifically their choice of verbs, tone, and formality based on hierarchical position, gender dynamics, and perceived relational closeness. Men will show more power-associated language than women, while senders will align their communication style with personas tied to their workplace role."

This hypothesis explores two key aspects:

- Hierarchical Influence: Employees will use more  deferential language when communicating “upward” in the corporate hierarchy and more casual, authoritative, or instructional language when speaking “downward”.

- Relational Closeness: Certain employees, will exhibit unique patterns of informality that set their communication apart from purely transactional work relationships (Steffensmeier et al., 2013).

####Methodological Contribution

While earlier studies laid the groundwork for understanding gender and power in communication, my work uses modern computational tools to streamline this kind of analysis. I use Riveter, an NLP toolkit that simplifies linguistic feature extraction such as identifying gendered language, sentiment, and tone shifts through verb-subject pairs. Riveter allows for consistent and scalable classification of linguistic patterns, replacing manual coding required in earlier research. Combined with large language models (LLMs) for text cleaning and classification, I believe that this approach will create a more generalized precedant that remains accurate, but is flexible to work with the different shapes of emails in this dataset. If this approach is successful, then when looking at other types of sender receiver networks, message isolation can be easily and adaptively changed just through prompting, and genderization on increasingly less look-up-able names like gamertags and usernames can be accomplished. This also could provide an accessible alternative that doesn't have to rely on social security lookups like in the Prabhakaran et al. study.

## 2. Data and methods


```python
#riveter reqires a lot of installs, and in order to save my work, I mounted drive
from google.colab import drive
drive.mount('/content/gdrive')
colab_directory_path = '/content/gdrive/My Drive/colab-output/riveter-test'
%cd /content/gdrive/My Drive/colab-output/riveter-test
! git clone -q https://github.com/maartensap/riveter-nlp.git
%ls
%cd /content/gdrive/My Drive/colab-output/riveter-test/riveter-nlp/riveter
!pip install -U -q spacy-experimental
!pip install -q https://github.com/explosion/spacy-experimental/releases/download/v0.6.0/en_coreference_web_trf-3.4.0a0-py3-none-any.whl#egg=en_coreference_web_trf
!python -m spacy download en_core_web_sm -q
!pip install -q seaborn
```

    Mounted at /content/gdrive
    /content/gdrive/My Drive/colab-output/riveter-test
    fatal: destination path 'riveter-nlp' already exists and is not an empty directory.
    [0m[01;34mriveter-nlp[0m/
    /content/gdrive/My Drive/colab-output/riveter-test/riveter-nlp/riveter
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m669.5/669.5 kB[0m [31m30.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m490.3/490.3 MB[0m [31m1.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m152.6/152.6 kB[0m [31m11.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m93.9/93.9 kB[0m [31m8.4 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.5/6.5 MB[0m [31m69.4 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.5/53.5 kB[0m [31m3.4 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m47.3/47.3 kB[0m [31m3.5 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.1/3.1 MB[0m [31m65.1 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m57.0/57.0 kB[0m [31m4.9 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m314.0/314.0 kB[0m [31m21.5 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m919.6/919.6 kB[0m [31m41.7 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.8/5.8 MB[0m [31m108.3 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.8/7.8 MB[0m [31m114.2 MB/s[0m eta [36m0:00:00[0m
    [?25h[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    albumentations 1.4.20 requires pydantic>=2.7.0, but you have pydantic 1.10.19 which is incompatible.
    en-core-web-sm 3.7.1 requires spacy<3.8.0,>=3.7.2, but you have spacy 3.4.4 which is incompatible.
    google-genai 0.2.2 requires pydantic<3.0.0dev,>=2.0.0, but you have pydantic 1.10.19 which is incompatible.
    langchain 0.3.11 requires pydantic<3.0.0,>=2.7.4, but you have pydantic 1.10.19 which is incompatible.
    langchain-core 0.3.24 requires pydantic<3.0.0,>=2.5.2; python_full_version < "3.12.4", but you have pydantic 1.10.19 which is incompatible.
    sentence-transformers 3.2.1 requires transformers<5.0.0,>=4.41.0, but you have transformers 4.25.1 which is incompatible.[0m[31m
    [0m2024-12-17 20:32:31.147514: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-12-17 20:32:31.180886: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-12-17 20:32:31.192571: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-12-17 20:32:32.949359: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1734467554.327586    1421 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
    I0000 00:00:1734467554.332496    1421 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
    I0000 00:00:1734467554.332681    1421 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m12.8/12.8 MB[0m [31m36.6 MB/s[0m eta [36m0:00:00[0m
    [?25h[38;5;2m✔ Download and installation successful[0m
    You can now load the package via spacy.load('en_core_web_sm')



```python
from collections import defaultdict
import pandas as pd
from riveter import Riveter #for getting riveter power dynamics
import seaborn as sns
import matplotlib.pyplot as plt

import pickle #for saving files
import kagglehub #pulls the dataset from kaggle
import numpy as np
import pandas as pd
import os
import re
import requests
from sqlalchemy import create_engine, text #merging title and name on email
import nltk #gender classification
from nltk.corpus import names
from tqdm import tqdm
tqdm.pandas()

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login #huggingface api key
import json
import math
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score #hyperparameter tuning on dbscan
from PIL import Image #image display
```

### Part 1: Data Cleaning


```python
path = kagglehub.dataset_download("wcukierski/enron-email-dataset")
print("Path to dataset files:", path)
file_path = os.path.join(path, 'emails.csv')
df = pd.read_csv(file_path)
```

    Warning: Looks like you're using an outdated `kagglehub` version, please consider updating (latest version: 0.3.5)
    Downloading from https://www.kaggle.com/api/v1/datasets/download/wcukierski/enron-email-dataset?dataset_version_number=2...


    100%|██████████| 358M/358M [00:11<00:00, 31.8MB/s]

    Extracting files...


    


    Path to dataset files: /root/.cache/kagglehub/datasets/wcukierski/enron-email-dataset/versions/2



```python
"""
Here we look at the shape of the data, and look at a sample message
"""
print(df.shape)
print(df.iloc[1000]['message'])
df.iloc[1000]['message']
```

    (517401, 2)
    Message-ID: <1081797.1075855696183.JavaMail.evans@thyme>
    Date: Thu, 15 Mar 2001 07:39:00 -0800 (PST)
    From: phillip.allen@enron.com
    To: stagecoachmama@hotmail.com
    Subject: 
    Mime-Version: 1.0
    Content-Type: text/plain; charset=us-ascii
    Content-Transfer-Encoding: 7bit
    X-From: Phillip K Allen
    X-To: stagecoachmama@hotmail.com
    X-cc: 
    X-bcc: 
    X-Folder: \Phillip_Allen_June2001\Notes Folders\All documents
    X-Origin: Allen-P
    X-FileName: pallen.nsf
    
    Lucy,
    
    Here is the rentroll.
    
    My only questions are about #18, #25, and #37 missed rent.  Any special 
    reasons?
    
    It looks like there are five vacancies #2,12,20a,35,40.  If you want to run 
    an ad in the paper with a $50 discount that is fine.
    I will write you a letter of recommendation.  When do you need it?  You can 
    use me as a reference.  In the next two weeks we should really have a good 
    idea whether the sale is going through.
    
    Phillip





    'Message-ID: <1081797.1075855696183.JavaMail.evans@thyme>\nDate: Thu, 15 Mar 2001 07:39:00 -0800 (PST)\nFrom: phillip.allen@enron.com\nTo: stagecoachmama@hotmail.com\nSubject: \nMime-Version: 1.0\nContent-Type: text/plain; charset=us-ascii\nContent-Transfer-Encoding: 7bit\nX-From: Phillip K Allen\nX-To: stagecoachmama@hotmail.com\nX-cc: \nX-bcc: \nX-Folder: \\Phillip_Allen_June2001\\Notes Folders\\All documents\nX-Origin: Allen-P\nX-FileName: pallen.nsf\n\nLucy,\n\nHere is the rentroll.\n\nMy only questions are about #18, #25, and #37 missed rent.  Any special \nreasons?\n\nIt looks like there are five vacancies #2,12,20a,35,40.  If you want to run \nan ad in the paper with a $50 discount that is fine.\nI will write you a letter of recommendation.  When do you need it?  You can \nuse me as a reference.  In the next two weeks we should really have a good \nidea whether the sale is going through.\n\nPhillip'



This particular message gives the indication to use regex to extract the message and metadata, so to get sender and receiver info through the emails, I took the text after the From and To. When looking at other messages of a different format (ie. diff subject lines or 1 liners), this strategy can't be used directly to get the sender and receiver name. Also, as can be seen in this email (stagecoachmama@hotmail.com), emails don't always follow a standard format.


```python
"""
This takes the sender and receiver, and extracts the email address
"""
def extract_sender(text):
  match = re.search(r"From: ([\w\.-]+@[\w\.-]+\.\w+)", text)
  if match:
    return match.group(1)
  return None
def extract_receiver(text):
  match = re.search(r"To: ([\w\.-]+@[\w\.-]+\.\w+)", text)
  if match:
    return match.group(1)
  return None
df['sender_email'] = df['message'].apply(extract_sender)
df['receiver_email'] = df['message'].apply(extract_receiver)
```


```python
# This code attempts to find the subject name using regex, however the names are too varied atm
# def fill_receiver_names(message):
#     """
#     Fills empty receiver names with Subject Name for a single message.
#     """
#     pattern = r"FileName: ([\w\.]+)\n\n([\w]+)\n"
#     match = re.search(pattern, message)
#     if match:
#         return match.group(2)  # Return the receiver name if found
#     return None  # Return None if no match

# Apply the modified function to extract receiver names
# df['receiver_name'] = df['message'].apply(fill_receiver_names)
# import nltk
# from nltk.corpus import names
# from tqdm import tqdm

# nltk.download('names')
# male_names = names.words('male.txt')
# female_names = names.words('female.txt')
# all_names = set(male_names + female_names)

# for name in tqdm(df.receiver_name):
#     if name == "jtholtski1122":
#     if name == None: continue
#     if name.capitalize() in all_names:
#       continue
#     else:
#         name = None
#         print(name)
# df.receiver_name.unique()

```

Above is code that attempts to extract the subject name with regex. It is commented out so that it doesn't bog down runtime when using the notebook, however as mentioned in the previous text block, the regex is inconsistent due to the many different ways of structuring the emails. The code below draws from a different preexisting dataset that maps full names and title onto the different emails. While this isn't a complete mapping, it provides a varied range of position types.


```python
"""
This code takes names and positions from an online source, so we can eventully
look at power and gender within the cleaned dataset. This source is relatively
incomplete, so I may use other methods like nltk and regex to fill in missing
names.
"""

url = "https://raw.githubusercontent.com/enrondata/enrondata/master/data/misc/edo_enron-custodians-data.tsv"
output_path = "edo_enron-custodians-data.tsv"
with open(output_path, 'wb') as f:
    f.write(requests.get(url).content)
names_df = pd.read_csv(output_path, sep='\t', header =None)
names_df.columns = ['folder', 'email', 'name', 'title']
names_df['email'] = names_df['email'].str.replace('philip.allen@enron.com', 'phillip.allen@enron.com')
names_df.head()
```





  <div id="df-66384b8d-69de-4858-97ed-1e86e7559df1" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>folder</th>
      <th>email</th>
      <th>name</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>allen-p</td>
      <td>phillip.allen@enron.com</td>
      <td>Philip Allen</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>arnold-j</td>
      <td>john.arnold@enron.com</td>
      <td>John Arnold</td>
      <td>Vice President</td>
    </tr>
    <tr>
      <th>2</th>
      <td>arora-h</td>
      <td>harpreet.arora@enron.com</td>
      <td>Harpreet Arora</td>
      <td>Vice President</td>
    </tr>
    <tr>
      <th>3</th>
      <td>badeer-r</td>
      <td>robert.badeer@enron.com</td>
      <td>Robert Badeer</td>
      <td>Director</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bailey-s</td>
      <td>susan.bailey@enron.com</td>
      <td>Susan Bailey</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-66384b8d-69de-4858-97ed-1e86e7559df1')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-66384b8d-69de-4858-97ed-1e86e7559df1 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-66384b8d-69de-4858-97ed-1e86e7559df1');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-07412f89-644f-4d4f-9920-b360dbad7e1a">
  <button class="colab-df-quickchart" onclick="quickchart('df-07412f89-644f-4d4f-9920-b360dbad7e1a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-07412f89-644f-4d4f-9920-b360dbad7e1a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
"""
This code merges the two data frames using SQLite
"""
#convert data frames to sql
engine = create_engine('sqlite:///:memory:')
df.to_sql('main', engine, if_exists='replace', index=False)
names_df.to_sql('names', engine, if_exists='replace', index=False)

#run query on data frames
query = text("""
SELECT main.*, n.name AS sender_name, n.title AS sender_title,
n2.name AS receiver_name, n2.title AS receiver_title
FROM main
LEFT JOIN names n ON main.sender_email = n.email
LEFT JOIN names n2 ON main.receiver_email = n2.email
""")

#Load final results into data frame and display
df = pd.read_sql_query(query, engine)
df.head()
```





  <div id="df-0700a040-8664-402d-8875-a32bf27cb565" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>message</th>
      <th>sender_email</th>
      <th>receiver_email</th>
      <th>sender_name</th>
      <th>sender_title</th>
      <th>receiver_name</th>
      <th>receiver_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>allen-p/_sent_mail/1.</td>
      <td>Message-ID: &lt;18782981.1075855378110.JavaMail.e...</td>
      <td>phillip.allen@enron.com</td>
      <td>tim.belden@enron.com</td>
      <td>Philip Allen</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>allen-p/_sent_mail/10.</td>
      <td>Message-ID: &lt;15464986.1075855378456.JavaMail.e...</td>
      <td>phillip.allen@enron.com</td>
      <td>john.lavorato@enron.com</td>
      <td>Philip Allen</td>
      <td>None</td>
      <td>John Lavorato</td>
      <td>CEO, Enron America</td>
    </tr>
    <tr>
      <th>2</th>
      <td>allen-p/_sent_mail/100.</td>
      <td>Message-ID: &lt;24216240.1075855687451.JavaMail.e...</td>
      <td>phillip.allen@enron.com</td>
      <td>leah.arsdall@enron.com</td>
      <td>Philip Allen</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>allen-p/_sent_mail/1000.</td>
      <td>Message-ID: &lt;13505866.1075863688222.JavaMail.e...</td>
      <td>phillip.allen@enron.com</td>
      <td>randall.gay@enron.com</td>
      <td>Philip Allen</td>
      <td>None</td>
      <td>Randall Gay</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>allen-p/_sent_mail/1001.</td>
      <td>Message-ID: &lt;30922949.1075863688243.JavaMail.e...</td>
      <td>phillip.allen@enron.com</td>
      <td>greg.piper@enron.com</td>
      <td>Philip Allen</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0700a040-8664-402d-8875-a32bf27cb565')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0700a040-8664-402d-8875-a32bf27cb565 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0700a040-8664-402d-8875-a32bf27cb565');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-cd4d59c8-9cd8-4ad0-b974-4bf2fe2dbe37">
  <button class="colab-df-quickchart" onclick="quickchart('df-cd4d59c8-9cd8-4ad0-b974-4bf2fe2dbe37')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-cd4d59c8-9cd8-4ad0-b974-4bf2fe2dbe37 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




The above data frame consists of the main object for quereying the rest of the project. With sql, I merged the names and titles on email and created sender and receiver metadata to be used later.


## Applying gender

The below code uses nltk to classify gender. This is similar to the approach used in Prabhakaran et. al, as the nltk implementation is much like their approach of looking at a list of legal names and identifying. I don't have access to social security records like they did, but what is important to note is the rigidity of this methodology, as well as looking to future projects with less common names ie. usernames or names uncommon in English


```python
"""
Using nltk to guess the genders for them. Other options include a manual
approach, or the genderizer.io api, but the api requires a key after use,
and I would prefer this methodology to be somewhat scalable and want to
limit manual inputting. I can also come back and use an approach that
applies confidence, which would help with potential bias from nltk.
"""
nltk.download('names')
male_names = set(name.lower() for name in names.words('male.txt'))
female_names = set(name.lower() for name in names.words('female.txt'))

"""
This function attempts to categorize by gender. nltk does have some negative,
but this should be fine for proof of concept.
"""
def classify_gender(name):
    if pd.isna(name) or not isinstance(name, str):
        return "unknown"
    name = name.lower()
    if name in male_names:
        return "male"
    elif name in female_names:
        return "female"
    else:
        return "unknown"

#split names by first and last and apply classify gender function (also add id)
#names splitting
df['sender_first_name'] = df['sender_name'].str.split().str[0]
df['receiver_first_name'] = df['receiver_name'].str.split().str[0]
df['sender_last_name'] = df['sender_name'].str.split().str[-1]
df['receiver_last_name'] = df['receiver_name'].str.split().str[-1]
#apply gender classification
df['sender_gender'] = df['sender_first_name'].apply(lambda name: classify_gender(name.lower()) if not pd.isna(name) else "unknown")
df['receiver_gender'] = df['receiver_first_name'].apply(lambda name: classify_gender(name.lower()) if not pd.isna(name) else "unknown")
#adding id column and filtered message for riveter stuff later
df['id'] = range(1, len(df) + 1)
df.head()
```

    [nltk_data] Downloading package names to /root/nltk_data...
    [nltk_data]   Unzipping corpora/names.zip.






  <div id="df-40ccb43c-435c-4cdb-a0f0-9aa65bf256e2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>message</th>
      <th>sender_email</th>
      <th>receiver_email</th>
      <th>sender_name</th>
      <th>sender_title</th>
      <th>receiver_name</th>
      <th>receiver_title</th>
      <th>sender_first_name</th>
      <th>receiver_first_name</th>
      <th>sender_last_name</th>
      <th>receiver_last_name</th>
      <th>sender_gender</th>
      <th>receiver_gender</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>allen-p/_sent_mail/1.</td>
      <td>Message-ID: &lt;18782981.1075855378110.JavaMail.e...</td>
      <td>phillip.allen@enron.com</td>
      <td>tim.belden@enron.com</td>
      <td>Philip Allen</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>Philip</td>
      <td>None</td>
      <td>Allen</td>
      <td>None</td>
      <td>male</td>
      <td>unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>allen-p/_sent_mail/10.</td>
      <td>Message-ID: &lt;15464986.1075855378456.JavaMail.e...</td>
      <td>phillip.allen@enron.com</td>
      <td>john.lavorato@enron.com</td>
      <td>Philip Allen</td>
      <td>None</td>
      <td>John Lavorato</td>
      <td>CEO, Enron America</td>
      <td>Philip</td>
      <td>John</td>
      <td>Allen</td>
      <td>Lavorato</td>
      <td>male</td>
      <td>male</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>allen-p/_sent_mail/100.</td>
      <td>Message-ID: &lt;24216240.1075855687451.JavaMail.e...</td>
      <td>phillip.allen@enron.com</td>
      <td>leah.arsdall@enron.com</td>
      <td>Philip Allen</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>Philip</td>
      <td>None</td>
      <td>Allen</td>
      <td>None</td>
      <td>male</td>
      <td>unknown</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>allen-p/_sent_mail/1000.</td>
      <td>Message-ID: &lt;13505866.1075863688222.JavaMail.e...</td>
      <td>phillip.allen@enron.com</td>
      <td>randall.gay@enron.com</td>
      <td>Philip Allen</td>
      <td>None</td>
      <td>Randall Gay</td>
      <td>None</td>
      <td>Philip</td>
      <td>Randall</td>
      <td>Allen</td>
      <td>Gay</td>
      <td>male</td>
      <td>male</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>allen-p/_sent_mail/1001.</td>
      <td>Message-ID: &lt;30922949.1075863688243.JavaMail.e...</td>
      <td>phillip.allen@enron.com</td>
      <td>greg.piper@enron.com</td>
      <td>Philip Allen</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>Philip</td>
      <td>None</td>
      <td>Allen</td>
      <td>None</td>
      <td>male</td>
      <td>unknown</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-40ccb43c-435c-4cdb-a0f0-9aa65bf256e2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-40ccb43c-435c-4cdb-a0f0-9aa65bf256e2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-40ccb43c-435c-4cdb-a0f0-9aa65bf256e2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-39c23789-4299-451a-a550-dabcae5b6058">
  <button class="colab-df-quickchart" onclick="quickchart('df-39c23789-4299-451a-a550-dabcae5b6058')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-39c23789-4299-451a-a550-dabcae5b6058 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
#This code creates a grouped bar chart to look at the distribution of gender
all_genders = set(df['sender_gender']).union(df['receiver_gender'])
sender_counts = df['sender_gender'].value_counts().reindex(all_genders, fill_value=0)
receiver_counts = df['receiver_gender'].value_counts().reindex(all_genders, fill_value=0)

x = np.arange(len(all_genders))
plt.figure(figsize=(4, 3))
plt.bar(x - 0.15, sender_counts, 0.3, label='Sender', color='blue')
plt.bar(x + 0.15, receiver_counts, 0.3, label='Receiver', color='orange')

plt.title('Gender Distribution for Sender and Receiver')
plt.xticks(x, all_genders)
plt.legend()
plt.tight_layout()
plt.show()

```


    
![png](clustering-on-gender-and-power_files/clustering-on-gender-and-power_18_0.png)
    


As we can see above, the nltk approach is still largely failing to recognize the names in the dataset. I decide to continue with this metric for now, however, since I don't have the compute to run the entire dataset anyway. Again, data that could directly map all of these names is much less accessible than nltk, so below I will use a language model to contextually determine gender.

Currently the messages contain a lot of metadata which will make tokenization difficult. As a part of this language model based approach, I will also use a model to trim metadata, so the tokens are only relaying the information from the main message body




# Part 2: LLM Approach to Gender Classification and Cleaning


```python
"""
I dont want to use too much of my colab gpu time so I will start by analyzing
riveter_df_mini, and then move onto riveter_df, which will contain all messages
that have gendered sender and receiver.
"""
#going to use these slices later for riveter so saving them under these names
riveter_df = df[(df['sender_gender'] != 'unknown') & (df['receiver_gender'] != 'unknown')]
riveter_df['id_gendered'] = range(1, len(riveter_df) + 1)
print(riveter_df.shape)
llm_df = riveter_df.copy()
```

    (23689, 16)


    <ipython-input-12-136d2b42dfdf>:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      riveter_df['id_gendered'] = range(1, len(riveter_df) + 1)



```python
"""
Using this code since I can't feed too many tokens into the model. I think that
having the metadata as context could be interesting to look at but Its not
going to be possible given the scope of this project.
"""
def truncate_message(text):
    pattern = r"X-FileName: .*?\n(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

llm_df['message_truncated'] = llm_df['message'].apply(truncate_message)

text_length = llm_df['message_truncated'].str.len()
text_length.plot(kind='hist',
                           bins=50,
                           edgecolor='black',
                           title='Text Length Distribution',
                           xlabel='Length',
                           figsize=(5, 3))
```




    <Axes: title={'center': 'Text Length Distribution'}, xlabel='Length', ylabel='Frequency'>




    
![png](clustering-on-gender-and-power_files/clustering-on-gender-and-power_22_1.png)
    


As can be seen by the figure above, some of the emails have very long lengths, which inhibits my strategy of using the llm to process the messages for information. While this could be 'solved' with a batching approach, that would remove a decent amount of context, which could hurt overall accuracy of the model. I am limited in compute and time limits on that compute, so I will demonstrate the project on a smaller length subset of the message data. Most of the messages fit within this range, meaning that if a model with a larger context window were used, it is likely that the speed wouldn't be effected too much given that only a small subset exceed the 10000 character range.


```python
#Only look at emails with smaller length
llm_df_mini = llm_df[llm_df['message'].apply(len) < 5000]
print(len(llm_df_mini))
llm_df_mini = llm_df_mini.iloc[:15]
```

    22119



```python
"""
This code uses an LLM to clean the metadata out of the messages, so they can be
better tokenized later in riveter and other EDA
"""
from google.colab import userdata

token = userdata.get("HF_TOKEN")
login(token=token)

# model_name = "EleutherAI/gpt-neo-1.3B" #larger takes 52 min on 100 rows w/gpu
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             use_auth_token=True,
                                             trust_remote_code=True,
                                             device_map="auto",
                                             torch_dtype="auto")
extract_pipeline = pipeline("text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            max_length=2048,
                            trust_remote_code=True)
device = model.device

"""
This function attempts to clean the message body using an LLM, since the
messages are all so varied.
"""
def extract_message_body(text):
    prompt = (
        "Extract the main message body (ignore metadata) from the following email:\n\n"
        f"{text}\n\n"
        "Message body:"
    )
    inputs = tokenizer(prompt,
                      return_tensors="pt",
                      padding=True,
                      truncation=True,
                      max_length=2048).to(device)
    output_ids = model.generate(inputs.input_ids,
                                attention_mask=inputs.attention_mask,
                                pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens=10,
                                do_sample=False
    )
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    result = result.split("Message body:")[-1].strip()
    return result

#apply functions
llm_df_mini['message_llm'] = llm_df_mini['message_truncated'].progress_apply(extract_message_body)
llm_df_mini['message_llm']

```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py:797: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(



    tokenizer_config.json:   0%|          | 0.00/727 [00:00<?, ?B/s]



    vocab.json:   0%|          | 0.00/899k [00:00<?, ?B/s]



    merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/2.11M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/357 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/1.01k [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/526M [00:00<?, ?B/s]


    100%|██████████| 15/15 [00:03<00:00,  4.20it/s]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>message_llm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>I would suggest holding the business plan meet...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Phillip,\n\nI have been</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Dave, \n\n Here are the names</td>
    </tr>
    <tr>
      <th>38</th>
      <td>John,\n\n Denver's short rockies</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Jim,\n\nI have been working</td>
    </tr>
    <tr>
      <th>48</th>
      <td>---------------------- Forwarded by Michael Et...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>---------------------- Forwarded by Phillip K ...</td>
    </tr>
    <tr>
      <th>53</th>
      <td>9/8  9/7</td>
    </tr>
    <tr>
      <th>65</th>
      <td>resumes of whom?\n\nMessage</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Cooper,\n \n Can you</td>
    </tr>
    <tr>
      <th>80</th>
      <td>---------------------- Forwarded by Kristian J</td>
    </tr>
    <tr>
      <th>94</th>
      <td>---------------------- Forwarded by Janel Guer...</td>
    </tr>
    <tr>
      <th>100</th>
      <td>I tried the new address but I don</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Cooper,\n\nI have a</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Cooper,\n\n Did you add some</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> object</label>



The above code takes a pretty long time with the resources that I am using right now. As I said in the comments, I went with the smaller 125M model. The messages tend to be pretty good, however, since much of this task is about filtering versus generating. The above output shows examples of the generated outputs, which were generally similar to the one full 22119 row run that I have saved to work with for the riveter and clustering.

While commented out to improve colab run all, this demo gender extracter was both expensive and largely ineffective. This error may have been due to the relatively small model being used, given that this task is much more difficult than the message filtering task.


```python
# model_name = "EleutherAI/gpt-neo-1.3B" #larger takes 52 min on 100 rows w/gpu
# # model_name = "EleutherAI/gpt-neo-125M"
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(model_name,
#                                              use_auth_token=True,
#                                              trust_remote_code=True,
#                                              device_map="auto",
#                                              torch_dtype="auto")
# extract_pipeline = pipeline("text-generation",
#                             model=model,
#                             tokenizer=tokenizer,
#                             max_length=2048,
#                             trust_remote_code=True)
# device = model.device

# """
# This function attempts to replace the function of nltk and the other cleaning
# that I found on the web by extracting the gender of the sender and receiver.
# If this approach works, well, then this could make similar tasks much easier,
# as a lot of analysis on this dataset is locked behind the data cleaning, and an
# LLM approach limits the need for regex and manual inputting.
# """
# def extract_receiver_gender(text):
#     prompt = (
#         "Read the following email and determine the gender of the receiver. "
#         "You must answer with exactly one word: 'male', 'female', or 'unknown'. "
#         "Do not include any additional text.\n\n"
#         f"Email:\n{text}\n\n"
#         "answer:"
#     )
#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=2048
#     ).to(device)

#     output_ids = model.generate(
#         inputs.input_ids,
#         attention_mask=inputs.attention_mask,
#         pad_token_id=tokenizer.pad_token_id,
#         max_new_tokens=10,
#         do_sample=False
#     )
#     response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()

#     matches = list(re.finditer(r'\b(male|female|unknown)\b', response))
#     if matches:
#         last_match = matches[-1]  # Last match (from the end)
#         return last_match.group(0)
#     else:
#         return 'unknown'  # Default to 'unknown' if no valid word is found

# llm_df_mini['receiver_gender_llm'] = llm_df_mini['message_llm'].progress_apply(extract_receiver_gender)
# print(llm_df_mini['receiver_gender_llm'].value_counts())
```


```python
#save the dataframe to a csv file
llm_df_mini.to_csv("####_llm_df_mini.csv", index=False)
```

### Message Cleaning --> success
- I felt that the message cleaning was overall a success. I successfully isolated the message body, even with a smaller model, and most of the uniqueness involved was isolated to prmopt engineering. The method also scaled to larger amounts of data, albiet requiring one of the colab gpus to run
- The results were slightly handheld by using the 'message_truncated' column instead of the flat message. While this did eliminate a lot of metadata, helping the task, it relies on my knowledge of the dataset. Part of this reason was to alleviate the smaller context window of the smaller model, but scaling up in model size did significantly increase runtime.
- Given a switch to larger models, a batching approach could be useful, and each call could clean multiple messages at once

### Gender Classification --> fail
- Gender classification, unfortunately, did not meet the same standard of nltk (even though it doesn't get everything). It had full coverage of the data, but in the two approaches that I tried, one analysing both sender and receiver and one doing both individually, The model failed to accurately guess gender most of the time. It defaulted to male for some of the more obvious answers, but even with limitations on the number of new tokens allowable, the model was fighting with the already truncated context window, and would often forget the prompting when answering
- While recommending a shot with a larger model is always an option, I think that either way some finetuning needs to be done to restrict the response more. Also, the issue of the 'unknown' gender category needs to be addressed, as given the examples that the model did output, they were only male labelled.

### LLM Methodology Results
- Using these models in an academic setting walks the line of accessibility. Like Professor said in class, the question of should I just buy a colab subscription for the course is an obstacle and a choice when applying text mining methods. On the other hand, the cleaning works, and I beleive that the gendering will work as well, given a larger model, with a larger context window and finetuning like we did in the bert labellings task homework. These approaches address my methodology goal of creating a more generalizable process. Much of the difference between these two functions is based in the prompting, which is easily changeable across tasks. Even with the compute restrictions, I still was able to get 22000 rows from the message cleaner. While this did take a significant amount of time, it was done on the free colab gpu. The Gender classifier stands as an argument against the scalability, as the extra finetuning work required to fix outputs to a desireable outcome could cost more compute than I used for the message proccessing. Particularly considering that this task only involved generation of a single additional token: "male", "female", "unknown"

# Part 3: Riveter Time!


```python
#I use this code to upload my saved riveter model
model_path = '/content/gdrive/MyDrive/colab-output/riveter-test/riveter-nlp/riveter/22000-length-1216-llm'
with open(model_path, 'rb') as file:
    riveter_llm = pickle.load(file)
print(riveter_llm)
```

    <riveter.Riveter object at 0x7a9119b96ef0>



```python
#use this code to train a new model

# #we use the filtered message so we dont get metadata in the training
# riveter_llm = Riveter()
# riveter_llm.load_sap_lexicon('power')
# riveter_llm.train(llm_df_mini['message_llm'], llm_df_mini['id'])

```


```python
#save the current riveter model
riveter_llm.save('#####-length-1216-llm')
```

    Riveter successfully saved to "#####-length-1216-llm"


Here I run the get_score_totals fucntion which aggregates scores for entities (personas) throughout a text dataset based on linkage frequency to verbs from riveters connection lexicon. Because I'm looking at aggregate power across title and gender, I set the frequency threshold parameter to one, which will compile all entities, and allow a more holistic analysis on the variety of subjects and direct objects witin the dataset.


```python
#shows all of the riveter scores calculated
riveter_scores = riveter_llm.get_score_totals(frequency_threshold=1)
riveter_scores
```




    {'i': 0.05986475735879077,
     'business plan meetings': -1.0,
     'cooper': 0.23529411764705882,
     'hunter': 0.0784313725490196,
     'west desk': -1.0,
     '2 analysts': 0.5714285714285714,
     'steve': -0.05806451612903226,
     'something': 0.0625,
     'jim:': -0.3333333333333333,
     'we': 0.1546572934973638,
     'andy:': 0.0,
     'way': -0.13333333333333333,
     'it': -0.1744186046511628,
     'message': 0.03787878787878788,
     'that feeling': 1.0,
     'p': 0.4117647058823529,
     'australia': 0.0,
     'little bit': -0.5,
     'stand alone button': 1.0,
     'ice': -1.0,
     'you': 0.1266607617360496,
     'deal': -0.5128205128205128,
     'mr sturm': 1.0,
     'lavor': 0.8,
     'bubbie': -1.0,
     'e': -0.14754098360655737,
     'restaurant': -1.0,
     'all the underdog - wouldn': 0.0,
     'him': 0.06666666666666667,
     'gaetien': 0.0,
     'task': -1.0,
     'anybody': -1.0,
     'four good tix': 1.0,
     'comments': -0.5087719298245614,
     '-----original message': -0.13333333333333333,
     'my': -0.18181818181818182,
     'tim': 0.0,
     'me': 0.10599078341013825,
     'report': 0.020833333333333332,
     'a': -0.0726643598615917,
     'email': 0.04032258064516129,
     'half a day': 0.0,
     'briefing': -1.0,
     'bob hall': -1.0,
     'stephanie': 0.041666666666666664,
     'kay': -0.04,
     'susan:': 0.0,
     'meeting': 0.056338028169014086,
     'copy': -0.2542372881355932,
     'two shits': -1.0,
     'friend': 0.5,
     "ya'll": -1.0,
     'another big name recruit': -1.0,
     'beasley': 1.0,
     'dues': 1.0,
     'grey': -1.0,
     'this': -0.03854875283446712,
     'wayne': -1.0,
     'matt': 0.0,
     'couple': -0.06666666666666667,
     'this e': 0.3076923076923077,
     '-': 0.0975609756097561,
     'mail': 0.09523809523809523,
     'item': 0.0,
     'atm': -1.0,
     'eric bass': 0.14285714285714285,
     'following table': 1.0,
     'seating chart': -1.0,
     'tom': 0.07142857142857142,
     'joe': 0.0,
     'you guys': -0.5,
     'lexi': 1.0,
     'record': 1.0,
     'that 127 employees': -1.0,
     'following names': -1.0,
     'mark frevert': 0.0,
     'attached addition': 0.0,
     'good progress': -1.0,
     'all': 0.16,
     'following message': -0.3125,
     'pre': 1.0,
     'tammie sch': -1.0,
     'jeff': 0.0,
     'the': -0.10224438902743142,
     'numbers': 1.0,
     'mike shanker': -1.0,
     'our': -0.48148148148148145,
     'mary solmonson': 0.14285714285714285,
     'ya': -0.5,
     'fyi': 0.006430868167202572,
     'shelley': -0.016129032258064516,
     'any': -0.17647058823529413,
     'lynn': -0.058823529411764705,
     'imperial holly corporation': 0.0,
     'this extension': 0.0,
     'board meeting': 1.0,
     'so sorry - hope things': 0.0,
     'following email': 0.011904761904761904,
     'net': -0.14285714285714285,
     'david port': 0.0,
     'she': 0.375,
     'v': -1.0,
     'either': 0.0,
     'anything': -0.6,
     'this message': 0.02631578947368421,
     'confidential information': -0.6666666666666666,
     'andrew': -0.4,
     'power trading': 1.0,
     'agreement': 0.16666666666666666,
     'john': -0.011235955056179775,
     'contract': -0.14285714285714285,
     'ricki ': 0.0,
     'this expense report': -1.0,
     'kim': -0.013333333333333334,
     'attached spreadsheet': -1.0,
     'corp security': 1.0,
     'quick audit': -1.0,
     'someone': 0.09090909090909091,
     'volumes': -1.0,
     'time': -0.022388059701492536,
     'chuckle': -1.0,
     'they': -0.4074074074074074,
     'add/drop': -1.0,
     'free agents': -0.4,
     'glynn': 1.0,
     'gap': 1.0,
     'call': -0.35714285714285715,
     'extension': -1.0,
     'joint resolution': 0.0,
     'approach': 0.0,
     'voicemail fro': -1.0,
     '-----original message-----': 0.007987220447284345,
     'delainey/whaley': 1.0,
     'no one': 0.6666666666666666,
     'her': 0.0,
     'presentation': -0.42105263157894735,
     'brattle': -1.0,
     'ppt': -1.0,
     'commercial teams': -1.0,
     'sites': 1.0,
     'attached file': 0.0,
     'file': -0.25,
     'fawcett': 1.0,
     'ford': -1.0,
     'he': 0.3333333333333333,
     'this material': -1.0,
     'gee': 1.0,
     'responses': -0.6,
     'ena': -0.09090909090909091,
     'deals': 0.3333333333333333,
     'cpuc': 0.16666666666666666,
     'px': 0.15384615384615385,
     'seabron': 0.3333333333333333,
     'draft': -0.037383177570093455,
     'some messages': -1.0,
     'points': -0.5,
     'ken': 0.18421052631578946,
     'msg': -1.0,
     'following points': 0.0,
     '': -1.0,
     'look': 0.0,
     'p.s': 0.0,
     'this information': 0.8571428571428571,
     'assembly energy committee': 1.0,
     'an': -0.45454545454545453,
     'wierdness': 0.0,
     'attached': 0.0,
     'assembly': 1.0,
     'budget': -1.0,
     'california': -0.017391304347826087,
     'mona': 1.0,
     'iso': 0.0,
     "kahn's secretary": 1.0,
     'messages': -0.4,
     'mary': 0.17857142857142858,
     'revisions': 0.21428571428571427,
     'kari': 1.0,
     'calif energy commission': 1.0,
     'jan': -0.08695652173913043,
     'jim': -0.007936507936507936,
     'michael tribolet': 0.5833333333333334,
     'conference': -1.0,
     'changed talking points': 0.0,
     'bob': -0.1111111111111111,
     'this version': 1.0,
     'this copy': 1.0,
     'issues': 0.0,
     'that number': -1.0,
     's': -0.5,
     'legal group': 1.0,
     'suit': -1.0,
     'eldon': -1.0,
     'mark': 0.08743169398907104,
     'letter': 0.10526315789473684,
     'reimbursement': 0.0,
     'greetings': 0.07142857142857142,
     'pg&e': -0.75,
     'sac bee': 1.0,
     'great minds': 0.0,
     'california puc': 1.0,
     'dwr': 0.0,
     "baldwin's secretary": -1.0,
     'average cost': -1.0,
     '/ac': -1.0,
     'while': 0.0,
     'bert': 0.5,
     'hard copies': -1.0,
     'that our allocation': 0.0,
     'lets': 0.2857142857142857,
     'rick which': 0.0,
     'rick buy': 0.5,
     'guys': 0.25,
     'pressure': -1.0,
     'scott': -0.12903225806451613,
     'swank': 0.0,
     'apathy': 1.0,
     'regards': 0.11764705882352941,
     'rick': 0.05263157894736842,
     'a) year end reviews - report': -1.0,
     'brian': 0.4,
     'one liners': 1.0,
     'barry': -0.027777777777777776,
     'order': -1.0,
     'any media questions': -1.0,
     'ena deals': 1.0,
     'answer': -0.5,
     'message body': -0.014084507042253521,
     'michael': 0.25,
     'slide': 0.0,
     'tracy geaccone': 0.09090909090909091,
     'vacation days': 0.0,
     'gina': -1.0,
     'slide #': -1.0,
     'amy': 0.0,
     'vacation': 0.0,
     'ron m': 1.0,
     'kevin': -0.013333333333333334,
     'products': -1.0,
     'tych': 0.0,
     'carmax': 1.0,
     'car': -0.5,
     'tino': 0.0,
     'open season posting': -1.0,
     'latest draft': 0.0,
     'if you': 0.4,
     "tino's consultant": 0.0,
     'whole thing': 0.0,
     'concern': -0.4666666666666667,
     'tw': 0.045454545454545456,
     'request': 0.1,
     'attached notes': 0.0,
     'situation': -0.4444444444444444,
     'susan': 0.016666666666666666,
     'some good': -1.0,
     'dave': 0.011235955056179775,
     'reaction': 1.0,
     "becky's draft answer": 0.0,
     'following purchase': -1.0,
     'signed copy': -1.0,
     'mk': 0.0,
     'this issue': 0.0,
     'original message': 0.0,
     "enron gov't affairs": 1.0,
     'this report': -1.0,
     'ferc': -0.0625,
     'protests': 0.6363636363636364,
     'laura': 0.0,
     'gallup contracts': -1.0,
     '10 minutes': -1.0,
     'them': -0.19230769230769232,
     'attached agreement': 0.0,
     'drafts': 0.0,
     'each': 0.2857142857142857,
     'canyou': 1.0,
     'detail': -0.6666666666666666,
     'run': 1.0,
     'danny': -0.16666666666666666,
     'this meeting': 0.3333333333333333,
     'office': -0.05263157894736842,
     'depreciation': -1.0,
     'everyone': -0.4166666666666667,
     'following draft': -1.0,
     'cindy/cng': 0.0,
     'judy townsend': -1.0,
     'robin': -1.0,
     'vickie': 1.0,
     'open season': -0.14285714285714285,
     'damon sykes': 1.0,
     'tp2': 1.0,
     'spinnaker gas': -1.0,
     'insufficient memory': -1.0,
     'access': -0.38461538461538464,
     'egan volumes': -1.0,
     'details': -0.5,
     'any gas': -1.0,
     '10,000 dth': -1.0,
     'txu': -0.3333333333333333,
     'enbridge': 0.0,
     'following payments': 0.0,
     'following changes': -1.0,
     'chris germany': 0.1,
     'new power company': 1.0,
     'demand': 1.0,
     'receipt': -1.0,
     'payment': 0.0,
     'prepayment': 0.0,
     'gas': -1.0,
     'melissa': 0.2857142857142857,
     'prepament': 0.0,
     'melissa garner': 1.0,
     'reliant': -1.0,
     '3,253 dth': -1.0,
     'ed': -0.07692307692307693,
     'job': 0.0,
     'following': -0.05660377358490566,
     'game': -0.17647058823529413,
     'chuck': -0.6,
     'maintenance': 0.0,
     'greg piper': 1.0,
     'following questions': -1.0,
     'mark golden': 1.0,
     'name': 0.5,
     'update': -0.125,
     'do-': 1.0,
     'julia': 0.0,
     'care': 0.0,
     'either you': -1.0,
     'policy': -0.3333333333333333,
     'slides': -1.0,
     'tucson': 0.0,
     'complaint': 0.0,
     'chart': -1.0,
     'feedback': 0.0,
     'dynegy': 0.0,
     'socal': 0.0,
     'udcs': -0.3333333333333333,
     'key provisions': 0.0,
     'long term transmission': -1.0,
     'verification': 1.0,
     "cal iso's order": 0.0,
     'attached letter': 0.0,
     'following list': -1.0,
     'diana ': 0.5,
     'nerc': -0.5,
     'following documents': 0.6,
     'personal achievement': -1.0,
     'rod': 0.0,
     'ets 2000 goals': -1.0,
     'gary': 0.3333333333333333,
     'case': -0.2222222222222222,
     'monica rodriquez': 0.0,
     'stan': 0.0,
     'fri': 0.0,
     'joe hunter': 0.0,
     'brother': 1.0,
     'dec': 0.0,
     'houston': -0.15,
     'red rock expansion': -1.0,
     'alot': -1.0,
     'john goodpasture': 1.0,
     'ca': 0.0,
     'transwestern negoti': 1.0,
     '"minimum': -1.0,
     'gisb': -0.3333333333333333,
     'language': -0.14285714285714285,
     'confirm': -0.25,
     'woman': 0.0,
     'status': -0.2222222222222222,
     'email address': -0.05714285714285714,
     'hard copy': -1.0,
     'new purchase agreement': 0.0,
     'brant reves': 0.0,
     'doc': 0.0,
     'master': -0.14285714285714285,
     'any addresses': 1.0,
     'collateral threshold amount': 0.0,
     '100% load factor': -1.0,
     'jason': 0.375,
     'box': 0.0,
     'database': -0.5,
     'nymex': 0.25,
     'martin': 0.0,
     '\n\nl': 0.0,
     'cheryl': -1.0,
     'lotus': -1.0,
     'problems': 1.0,
     'javascript': -1.0,
     'notebook': -1.0,
     'consolidated': -1.0,
     'several brokerage agreements': -1.0,
     'countries': -1.0,
     'these pages': 0.0,
     'tanya': -0.25,
     'power desk': 1.0,
     'mark taylor': -0.08695652173913043,
     'haedicke': 1.0,
     'tana jones': 0.05263157894736842,
     'silent treatment': -1.0,
     'form': -0.18181818181818182,
     'memo': 0.08695652173913043,
     'chill': -1.0,
     'laurel adams': 0.0,
     'desk': 0.0,
     'london legal': -1.0,
     'nothing': -0.4,
     "edward's markup": -1.0,
     'anthony': 0.0,
     'master distribution list': -1.0,
     'bill bradford': -1.0,
     'tana:': 0.08333333333333333,
     'any contracts': -0.5,
     'master agreement': 1.0,
     'john arnold': 0.3333333333333333,
     'shankman': 1.0,
     'bob shults': -0.14285714285714285,
     'attached message body': -1.0,
     'leslie hansen': -0.6666666666666666,
     'tana': 0.045454545454545456,
     'fewer resolutions': -1.0,
     'text': -0.5,
     'valid email address': -1.0,
     'copies': -0.5,
     'system': 1.0,
     'gold star': -1.0,
     'outstanding guaranties': -1.0,
     'track': -1.0,
     '7 file': 1.0,
     'san diego': 0.6666666666666666,
     'contract administration': 1.0,
     'following information': -0.75,
     '"forms': -1.0,
     'scoop': -1.0,
     'edmund cooper': -1.0,
     'cert': -1.0,
     'list': -0.043478260869565216,
     'updated version': -1.0,
     'inquiry': 1.0,
     'sand': 0.0,
     'nadina henley': 1.0,
     'louise': 0.007194244604316547,
     'attached bullets': 1.0,
     'real-life examples': -1.0,
     'richard shapiro': 0.0,
     'james lam': 0.0,
     'results': -1.0,
     'word': 0.7142857142857143,
     'pg&e corp': 0.0,
     'maureen raymond@ect': 1.0,
     'three documents': 0.0,
     'michelle ': 0.0,
     'palmisano': 1.0,
     'envelope': -1.0,
     'group': -0.1111111111111111,
     'message subject': -0.3333333333333333,
     'netco conference': -1.0,
     'clueless wonder': -1.0,
     'information': -0.029411764705882353,
     '24 new associates': -1.0,
     'willis phillips': -1.0,
     'brawner': 1.0,
     'presto': 0.0,
     'good people': -1.0,
     "pg&e's filing": 0.1111111111111111,
     'louise\n \ni': 0.0,
     'column': -1.0,
     'governance schedule': -1.0,
     'offer letter': 0.0,
     'conference call': -0.8,
     'arnold': 0.16666666666666666,
     'this group': -1.0,
     'port': -1.0,
     'estate / buyer': 1.0,
     'howard': -0.6666666666666666,
     'calger': 0.5,
     'likes': -1.0,
     'david': 0.16666666666666666,
     'wharton': 1.0,
     'greg': 0.0,
     'national fuel marketing': -1.0,
     'hey martin ': 0.3333333333333333,
     'all files': -1.0,
     '5 points': -1.0,
     '10 points': -1.0,
     'data': 1.0,
     '45 million var': -1.0,
     'overdyke': 0.0,
     'resumes': 0.0,
     'dave\n\n': -0.5,
     'ppa auction': 0.0,
     'rick\n\n': 0.2857142857142857,
     'more detail': 0.0,
     'temporary increase': 0.0,
     'crude group': 1.0,
     'this memo': -1.0,
     '=': -0.3333333333333333,
     'ees': 0.5,
     'pc': -1.0,
     'current': -1.0,
     'phone interview': -1.0,
     'denver +3': -1.0,
     'all the extra var': 1.0,
     'this email': -0.23076923076923078,
     'redmond': -1.0,
     'andy': -0.014925373134328358,
     'lap dance': -1.0,
     'stephen': 0.0,
     'd': 0.0,
     'that tropical storm': 1.0,
     'that bet': 0.0,
     'rockies': -1.0,
     'forecast': -0.42857142857142855,
     'jeff skilling': 1.0,
     'rivalry': 1.0,
     'writing': 0.0,
     'few changes': -1.0,
     'bachelor party': 1.0,
     'chance': -0.25,
     'that shit\n\nmessage': -1.0,
     'phillip': -0.16666666666666666,
     'elizabeth': 0.08108108108108109,
     'michelle': -0.1,
     'kn': -1.0,
     'eog': 1.0,
     'interconnect': -1.0,
     'commission': 0.75,
     'draft order': -1.0,
     'strip': 1.0,
     'changes': 0.0,
     '"cheat sheet': -1.0,
     'incremental revenue': -1.0,
     'thanks': 0.003978779840848806,
     'this posting': 0.0,
     'williams field services': 0.0,
     'new o': 0.0,
     'burlington': 0.2857142857142857,
     'framework': -1.0,
     'capability': 0.0,
     'mhq': -1.0,
     'operational gas': -1.0,
     'cirque du': 1.0,
     '5,000 mmbtu/': -1.0,
     'total': -1.0,
     '10,000 mmbtu': -1.0,
     'incremental 18,000/d firm': -1.0,
     'oneok': 0.2,
     'rofr': -1.0,
     'earl chanley': 1.0,
     'following two files': -1.0,
     'draft \n\n': 0.5,
     'not so fast buddy': 1.0,
     'favor': -1.0,
     'internal agreement': 1.0,
     'one or more contract questions': -1.0,
     '900am': 1.0,
     'a:': 0.0,
     'clarksdale': 0.0,
     'notice': 1.0,
     'this stuff': -0.5,
     'problem': -0.005714285714285714,
     'amount': -0.5,
     'unprecedented number': 1.0,
     'eric': -0.17647058823529413,
     'attachment': -0.0967741935483871,
     'o': -0.5,
     'your': 0.047619047619047616,
     '"e': -1.0,
     'ticket': 0.0,
     "i'm": 1.0,
     'books': 0.0,
     '2500/d': -1.0,
     'kam': 0.3333333333333333,
     'brad': -1.0,
     'this origination': -1.0,
     'prices': -1.0,
     'internet explorer': -1.0,
     'realmoney.com': -1.0,
     'half day': 0.0,
     'th contract #': -1.0,
     'global database': -1.0,
     'thoughts': -0.6666666666666666,
     'attached credit sheet': -1.0,
     'one page': 1.0,
     'aec': 0.0,
     'david e deutsch': 0.0,
     'response': 0.0,
     'pap': 0.0,
     'gerald': 0.16071428571428573,
     'contract price': -1.0,
     'julie': 0.6666666666666666,
     'mid month': 0.0,
     'debra': 0.038461538461538464,
     'documents': -0.5,
     'palo alto': -1.0,
     'msa': 1.0,
     'few tweaks': -1.0,
     'revised devon': 0.0,
     'clean documents': -1.0,
     'one more thing': -1.0,
     'tentative conference call': -1.0,
     'economics sheet': 0.0,
     'some': -0.35714285714285715,
     'revised net': -1.0,
     'huber transaction agreement': -1.0,
     'term': -1.0,
     'attached revision': 0.0,
     'this draft': -0.5,
     'final look': 0.0,
     'attached confirm': 0.0,
     'enron lng': 0.0,
     'link': 0.0,
     'sarah pyles': 1.0,
     "jennifer fraser's memo": 0.0,
     'dogs': 1.0,
     'termination notice': 1.0,
     'ed baughman': 1.0,
     'prudency c summaries': 1.0,
     '6 analysts': 1.0,
     'this guy': 0.16666666666666666,
     'map': -1.0,
     'jon': 1.0,
     'cinergy': 0.0,
     'kevin \ni': 0.0,
     'epmi': 1.0,
     'format': 0.25,
     'test': 0.25,
     "i'll": 1.0,
     'david w delainey': 0.08,
     'eob': -1.0,
     'enron europe': 0.2,
     'mark woolfolk\n\n': 0.5,
     'nancy pickover': 0.0,
     'weight': -1.0,
     'spreadsheet': -0.2,
     'darrell': -0.16666666666666666,
     "bp's meeting": 1.0,
     'question': 0.006711409395973154,
     'ambrosia': 1.0,
     'this document': 1.0,
     'both the el paso': -1.0,
     'obsession': -1.0,
     'incident': 1.0,
     'ted': -0.6666666666666666,
     'one': 1.0,
     'new isda': -1.0,
     'kim ward': 0.16666666666666666,
     'cheryl nelson': 0.2,
     "tana's list": 0.0,
     'veronica valdez': -1.0,
     'melissa murphy': -1.0,
     'omnibus': 0.0,
     'tracy ngo': 0.15384615384615385,
     'credit': -0.5454545454545454,
     'marie': 0.0,
     'help': 0.021052631578947368,
     'david vitrella': -1.0,
     'signature block': -1.0,
     'voice mail': -0.6923076923076923,
     'susan f': 0.0,
     'this e-mail message': 0.7333333333333333,
     '5 pm flight': -1.0,
     'steve morus': -1.0,
     'guaranty': 0.0,
     'russell diamond': 0.0,
     'boilerplate isda master agreement': -1.0,
     'who jane wilhite': -1.0,
     'mg ltd': 1.0,
     'finance': 1.0,
     'rodney/finley/t': -1.0,
     'us': 0.25,
     'marcie youngerman': -1.0,
     'julian poole': 0.5,
     'master" section': -1.0,
     'ect': 0.5,
     'lotus notes': 0.2,
     'tiger': -0.4,
     'counterparty': 0.25,
     'ena signature': 1.0,
     'any questions': -1.0,
     'what lawyer': 0.0,
     'east and west power desks': 1.0,
     'daily termination list': 1.0,
     'anyone': 0.0,
     'mahonia limited file': 0.0,
     'confirm desk': 0.0,
     'master log': 1.0,
     'embedded image': 1.0,
     'mcmahon': 1.0,
     'mike': -0.13043478260869565,
     'argentine crisis': 1.0,
     'specter': -1.0,
     'filings': 0.0,
     'skilling': 0.5,
     'this interview': -1.0,
     'transmission deals': 0.0,
     'confirm letters': -1.0,
     'leg': -1.0,
     'two smurfit deals': -1.0,
     'puget sound': 0.0,
     'high level': 1.0,
     'assistance': -1.0,
     'billing': -1.0,
     'going': 1.0,
     'any brokers': -1.0,
     'new markets/products': 1.0,
     'enron': 0.0,
     'koch': 1.0,
     'few minutes': 0.0,
     'these forms': -1.0,
     'any master agreements': -1.0,
     'mark h': -0.5,
     'sara': 0.020833333333333332,
     'andrea cal': 0.0,
     'phone call': 1.0,
     'this opinion': 0.0,
     'view': 0.4,
     'end': 0.0,
     'following note': 0.0,
     'any concerns': -1.0,
     'transcolorado': -1.0,
     'independent': 1.0,
     'some names': -1.0,
     'xmas cards': -1.0,
     'gag gift exchange': -1.0,
     'this sheet': 1.0,
     'temperature model': -1.0,
     'kevin lambert': 0.0,
     'ozzie': 0.0,
     'discussions': 0.3333333333333333,
     'lavo': 0.07692307692307693,
     'meeting plans': 1.0,
     'gift certificate': -1.0,
     'credit worksheet': -0.14285714285714285,
     'good place - bay area': 0.0,
     'check': -1.0,
     'lunch': 0.1111111111111111,
     'ranelle': 0.0,
     'attached interconnect agreement': 0.0,
     'bill': -0.04,
     'lindy': -0.02127659574468085,
     'man': 0.25,
     'jay': 0.2,
     'lisa - \ni': 0.0,
     'emails': -0.5,
     'reuters': 1.0,
     'louise\n\n\n\nespeed': 1.0,
     '75%': -1.0,
     'apr': -1.0,
     'errol': -0.5,
     'errol pleae': 1.0,
     'buddy': 1.0}




```python
#two functions as defined in the riveter colab demo, to look at subject power
persona_polarity_verb_count_dict = riveter_llm.get_persona_polarity_verb_count_dict()
print(len(persona_polarity_verb_count_dict))
persona_score_dict = riveter_llm.get_score_totals(frequency_threshold=20)
print(len(persona_score_dict))
```

    597
    26


The below code iterates through the generated personas in riveter, which aggregate total verb matchings on a subject's participation to assign their power score.


```python
#this looks at the generated personas, I found it only to be useful at 10k+ rows
for _persona, _score in sorted(persona_score_dict.items(), key=lambda x: x[1], reverse=True):
    print(round(_score, 2), '\t', _persona)
```

    0.15 	 we
    0.13 	 you
    0.11 	 me
    0.09 	 mark
    0.06 	 i
    0.06 	 meeting
    0.04 	 email
    0.04 	 message
    0.01 	 fyi
    0.0 	 jeff
    0.0 	 attached file
    0.0 	 following points
    0.0 	 look
    0.0 	 response
    -0.01 	 john
    -0.01 	 kim
    -0.04 	 this
    -0.07 	 a
    -0.1 	 the
    -0.17 	 it
    -0.19 	 them
    -0.25 	 copy
    -0.36 	 call
    -0.51 	 comments
    -0.51 	 deal
    -0.6 	 anything



```python
#looking into the verbs that are defining the power for 'jeff'
persona_polarity_verb_count_dict['jeff']
```




    defaultdict(<function riveter.default_dict_int()>,
                {'negative': defaultdict(int, {'need_nsubj': 4, 'want_nsubj': 4}),
                 'positive': defaultdict(int,
                             {'check_nsubj': 2,
                              'receive_dobj': 2,
                              'attach_nsubj': 4})})



In the example above, we see that the persona for 'Jeff' is influenced by this profile of verbs


```python
"""
Having used riveter as a metric of power on the cleaned emails, I want to also
apply a metric of gender onto the dataset. While my initial approach to use a
model to get this outran the scope of my project, I'll be using the free namesor
api to get a confidence score and labelled gender
"""
def get_gender_namsor_batch(names):
    """
    Send a batch of names to NamSor API and return their gender probability
    scores. Process in batches of 99 names.
    """
    namesor_API_KEY = userdata.get("NM_TOKEN")
    url = "https://v2.namsor.com/NamSorAPIv2/api2/json/genderBatch"
    headers = {"X-API-KEY": namesor_API_KEY, "Content-Type": "application/json"}

    gender_scores = {}

    #using batch processing since api costs per pull
    num_batches = math.ceil(len(names) / 99)
    for batch_idx in range(num_batches):
        batch_names = names[batch_idx * 99:(batch_idx + 1) * 99]
        payload = {"personalNames": [
            {"id": str(i), "firstName": name, "lastName": ""}
            for i, name in enumerate(batch_names)
        ]}

        response = requests.post(url, json=payload, headers=headers)
        data = response.json()

        #gets the probability scores from the genders it found
        for result in data.get("personalNames", []):
            name = result.get("firstName")
            gender = result.get("likelyGender", "unknown")
            probability = result.get("probabilityCalibrated", 0.0)

            if gender == "female":
                probability = -probability
            elif gender == "male":
                probability = probability
            else:
                probability = 0.0

            gender_scores[name] = probability
    return gender_scores
```


```python
namesor_API_KEY = userdata.get("NM_TOKEN")
def get_gender_namsor_batch(names):
    url = "https://v2.namsor.com/NamSorAPIv2/api2/json/genderBatch"
    headers = {"X-API-KEY": namesor_API_KEY, "Content-Type": "application/json"}

    payload = {
        "personalNames": [{"id": str(i),
                           "firstName": name,
                           "lastName": ""} for i, name in enumerate(names)]
    }
    response = requests.post(url, json=payload, headers=headers)
    data = response.json()

    results = {}
    if "personalNames" in data:
        for result in data["personalNames"]:
            name = result.get("firstName", "unknown")
            likely_gender = result.get("likelyGender", "unknown")
            probability_calibrated = result.get("probabilityCalibrated", 0.0)
            if likely_gender == "male":
                results[name] = probability_calibrated
            elif likely_gender == "female":
                results[name] = -probability_calibrated
            else:
                results[name] = 0.0
    return results

# Test
test_names = ["Riley", "Alex", "Sarah"]
gender_scores = get_gender_namsor_batch(test_names)
print(gender_scores)
```

    {'Riley': 0.7579952447825479, 'Alex': 0.9750108404939797, 'Sarah': -0.9830545378641277}


Given the riveter model trained on the data, I have a metric of power ranging from -1 to 1 depending on the entities interaction throughout the corpus. To analayze this categorization in a gendered context, I need a way of categorizing gender. Because I concluded that the llm prompting approach did not fit the goals of this project, I instead used namesor.com genderizing api. While the free plan only has limited api calls, I can use batching of inputted names to get more calls per day. These scores are also useful over the traditional nltk approach, as they give a gender confidence level, which theoretically would be helpful for analyzing pronouns, less common names, and could likely work for classifying usernames as discussed earlier.


```python
#processing to get ready for graphing and so I can save the gendered results
riveter_names = list(riveter_scores.keys())
power_scores = list(riveter_scores.values())

gender_scores = get_gender_namsor_batch(riveter_names)

data = {
    'name': riveter_names,
    'power_score': power_scores,
    'gender_score': [gender_scores.get(name, 0.0) for name in riveter_names]
}

gender_power_df = pd.DataFrame(data)
```


```python
#save current iteration of gender_power_df
gender_power_df.to_csv("gender_power_df_####.csv", index=False)
```

The below code plots power and gender score on a scatter plot, allowing me to analyze their potential relationship, as well as entity similarities 'geographically'. I will also generate boxplots to be able to get a good read on the difference between male and female power within this analyzed subset of the data. I grouped with 3 confidence levels high, low and all. High includes scores above the 95% threshold, which is primarily capturing names. Medium captures all names, but includes much more noise. Lastly the all confidence plots all datapoints for comparison.


```python
#Create a scatterplot showing gender score versus power score
fig = px.scatter(
    gender_power_df,
    x="power_score",
    y="gender_score",
    hover_data=["name", "power_score", "gender_score"],
    title="Power vs Gender Scores",
    labels={"power_score": "Power Score", "gender_score": "Gender Score"},
)
fig.update_xaxes(title="Power Score", zeroline=True)
fig.update_yaxes(title="Gender Score", zeroline=True)
fig.add_hline(y=0, line=dict(color="black", width=1))  # Horizontal Line
fig.add_vline(x=0, line=dict(color="black", width=1))  # Vertical Line
fig.show()

#Leverage gender confidnence, to understand potential correlation
high_conf_df = gender_power_df[gender_power_df['gender_score'].abs() > 0.95]
mid_conf_df = gender_power_df[(gender_power_df['gender_score'].abs() > 0.75)]
all_conf_df = gender_power_df

conf_dfs = {
    'High Confidence (>0.95)': high_conf_df,
    'Mid Confidence (>0.8)': mid_conf_df,
    'All Confidence (>0)': all_conf_df
}

fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
fig.suptitle("Power Scores for Male and Female Words by Confidence Levels", fontsize=16)

for ax, (title, df) in zip(axes, conf_dfs.items()):
    male_words = df.loc[df['gender_score'] > 0].copy()
    female_words = df.loc[df['gender_score'] < 0].copy()
    male_words['label'] = 'Male Words'
    female_words['label'] = 'Female Words'

    stacked_df = pd.concat([male_words, female_words])
    sns.boxplot(x='label', y='power_score', data=stacked_df, ax=ax,
                palette={'Male Words': 'blue', 'Female Words': 'red'}, hue='label', legend=False)
    ax.set_title(title)
    ax.set_xlabel("Gender")
    ax.set_ylabel("Power Score" if ax == axes[0] else "")
plt.tight_layout()
plt.show()


```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="b9df4588-235c-4886-a5d1-859813c1eb2e" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("b9df4588-235c-4886-a5d1-859813c1eb2e")) {                    Plotly.newPlot(                        "b9df4588-235c-4886-a5d1-859813c1eb2e",                        [{"customdata":[["i"],["business plan meetings"],["cooper"],["hunter"],["west desk"],["2 analysts"],["steve"],["something"],["jim:"],["we"],["andy:"],["way"],["it"],["message"],["that feeling"],["p"],["australia"],["little bit"],["stand alone button"],["ice"],["you"],["deal"],["mr sturm"],["lavor"],["bubbie"],["e"],["restaurant"],["all the underdog - wouldn"],["him"],["gaetien"],["task"],["anybody"],["four good tix"],["comments"],["-----original message"],["my"],["tim"],["me"],["report"],["a"],["email"],["half a day"],["briefing"],["bob hall"],["stephanie"],["kay"],["susan:"],["meeting"],["copy"],["two shits"],["friend"],["ya'll"],["another big name recruit"],["beasley"],["dues"],["grey"],["this"],["wayne"],["matt"],["couple"],["this e"],["-"],["mail"],["item"],["atm"],["eric bass"],["following table"],["seating chart"],["tom"],["joe"],["you guys"],["lexi"],["record"],["that 127 employees"],["following names"],["mark frevert"],["attached addition"],["good progress"],["all"],["following message"],["pre"],["tammie sch"],["jeff"],["the"],["numbers"],["mike shanker"],["our"],["mary solmonson"],["ya"],["fyi"],["shelley"],["any"],["lynn"],["imperial holly corporation"],["this extension"],["board meeting"],["so sorry - hope things"],["following email"],["net"],["david port"],["she"],["v"],["either"],["anything"],["this message"],["confidential information"],["andrew"],["power trading"],["agreement"],["john"],["contract"],["ricki "],["this expense report"],["kim"],["attached spreadsheet"],["corp security"],["quick audit"],["someone"],["volumes"],["time"],["chuckle"],["they"],["add\u002fdrop"],["free agents"],["glynn"],["gap"],["call"],["extension"],["joint resolution"],["approach"],["voicemail fro"],["-----original message-----"],["delainey\u002fwhaley"],["no one"],["her"],["presentation"],["brattle"],["ppt"],["commercial teams"],["sites"],["attached file"],["file"],["fawcett"],["ford"],["he"],["this material"],["gee"],["responses"],["ena"],["deals"],["cpuc"],["px"],["seabron"],["draft"],["some messages"],["points"],["ken"],["msg"],["following points"],[""],["look"],["p.s"],["this information"],["assembly energy committee"],["an"],["wierdness"],["attached"],["assembly"],["budget"],["california"],["mona"],["iso"],["kahn's secretary"],["messages"],["mary"],["revisions"],["kari"],["calif energy commission"],["jan"],["jim"],["michael tribolet"],["conference"],["changed talking points"],["bob"],["this version"],["this copy"],["issues"],["that number"],["s"],["legal group"],["suit"],["eldon"],["mark"],["letter"],["reimbursement"],["greetings"],["pg&e"],["sac bee"],["great minds"],["california puc"],["dwr"],["baldwin's secretary"],["average cost"],["\u002fac"],["while"],["bert"],["hard copies"],["that our allocation"],["lets"],["rick which"],["rick buy"],["guys"],["pressure"],["scott"],["swank"],["apathy"],["regards"],["rick"],["a) year end reviews - report"],["brian"],["one liners"],["barry"],["order"],["any media questions"],["ena deals"],["answer"],["message body"],["michael"],["slide"],["tracy geaccone"],["vacation days"],["gina"],["slide #"],["amy"],["vacation"],["ron m"],["kevin"],["products"],["tych"],["carmax"],["car"],["tino"],["open season posting"],["latest draft"],["if you"],["tino's consultant"],["whole thing"],["concern"],["tw"],["request"],["attached notes"],["situation"],["susan"],["some good"],["dave"],["reaction"],["becky's draft answer"],["following purchase"],["signed copy"],["mk"],["this issue"],["original message"],["enron gov't affairs"],["this report"],["ferc"],["protests"],["laura"],["gallup contracts"],["10 minutes"],["them"],["attached agreement"],["drafts"],["each"],["canyou"],["detail"],["run"],["danny"],["this meeting"],["office"],["depreciation"],["everyone"],["following draft"],["cindy\u002fcng"],["judy townsend"],["robin"],["vickie"],["open season"],["damon sykes"],["tp2"],["spinnaker gas"],["insufficient memory"],["access"],["egan volumes"],["details"],["any gas"],["10,000 dth"],["txu"],["enbridge"],["following payments"],["following changes"],["chris germany"],["new power company"],["demand"],["receipt"],["payment"],["prepayment"],["gas"],["melissa"],["prepament"],["melissa garner"],["reliant"],["3,253 dth"],["ed"],["job"],["following"],["game"],["chuck"],["maintenance"],["greg piper"],["following questions"],["mark golden"],["name"],["update"],["do-"],["julia"],["care"],["either you"],["policy"],["slides"],["tucson"],["complaint"],["chart"],["feedback"],["dynegy"],["socal"],["udcs"],["key provisions"],["long term transmission"],["verification"],["cal iso's order"],["attached letter"],["following list"],["diana "],["nerc"],["following documents"],["personal achievement"],["rod"],["ets 2000 goals"],["gary"],["case"],["monica rodriquez"],["stan"],["fri"],["joe hunter"],["brother"],["dec"],["houston"],["red rock expansion"],["alot"],["john goodpasture"],["ca"],["transwestern negoti"],["\"minimum"],["gisb"],["language"],["confirm"],["woman"],["status"],["email address"],["hard copy"],["new purchase agreement"],["brant reves"],["doc"],["master"],["any addresses"],["collateral threshold amount"],["100% load factor"],["jason"],["box"],["database"],["nymex"],["martin"],["\n\nl"],["cheryl"],["lotus"],["problems"],["javascript"],["notebook"],["consolidated"],["several brokerage agreements"],["countries"],["these pages"],["tanya"],["power desk"],["mark taylor"],["haedicke"],["tana jones"],["silent treatment"],["form"],["memo"],["chill"],["laurel adams"],["desk"],["london legal"],["nothing"],["edward's markup"],["anthony"],["master distribution list"],["bill bradford"],["tana:"],["any contracts"],["master agreement"],["john arnold"],["shankman"],["bob shults"],["attached message body"],["leslie hansen"],["tana"],["fewer resolutions"],["text"],["valid email address"],["copies"],["system"],["gold star"],["outstanding guaranties"],["track"],["7 file"],["san diego"],["contract administration"],["following information"],["\"forms"],["scoop"],["edmund cooper"],["cert"],["list"],["updated version"],["inquiry"],["sand"],["nadina henley"],["louise"],["attached bullets"],["real-life examples"],["richard shapiro"],["james lam"],["results"],["word"],["pg&e corp"],["maureen raymond@ect"],["three documents"],["michelle "],["palmisano"],["envelope"],["group"],["message subject"],["netco conference"],["clueless wonder"],["information"],["24 new associates"],["willis phillips"],["brawner"],["presto"],["good people"],["pg&e's filing"],["louise\n \ni"],["column"],["governance schedule"],["offer letter"],["conference call"],["arnold"],["this group"],["port"],["estate \u002f buyer"],["howard"],["calger"],["likes"],["david"],["wharton"],["greg"],["national fuel marketing"],["hey martin "],["all files"],["5 points"],["10 points"],["data"],["45 million var"],["overdyke"],["resumes"],["dave\n\n"],["ppa auction"],["rick\n\n"],["more detail"],["temporary increase"],["crude group"],["this memo"],["="],["ees"],["pc"],["current"],["phone interview"],["denver +3"],["all the extra var"],["this email"],["redmond"],["andy"],["lap dance"],["stephen"],["d"],["that tropical storm"],["that bet"],["rockies"],["forecast"],["jeff skilling"],["rivalry"],["writing"],["few changes"],["bachelor party"],["chance"],["that shit\n\nmessage"],["phillip"],["elizabeth"],["michelle"],["kn"],["eog"],["interconnect"],["commission"],["draft order"],["strip"],["changes"],["\"cheat sheet"],["incremental revenue"],["thanks"],["this posting"],["williams field services"],["new o"],["burlington"],["framework"],["capability"],["mhq"],["operational gas"],["cirque du"],["5,000 mmbtu\u002f"],["total"],["10,000 mmbtu"],["incremental 18,000\u002fd firm"],["oneok"],["rofr"],["earl chanley"],["following two files"],["draft \n\n"],["not so fast buddy"],["favor"],["internal agreement"],["one or more contract questions"],["900am"],["a:"],["clarksdale"],["notice"],["this stuff"],["problem"],["amount"],["unprecedented number"],["eric"],["attachment"],["o"],["your"],["\"e"],["ticket"],["i'm"],["books"],["2500\u002fd"],["kam"],["brad"],["this origination"],["prices"],["internet explorer"],["realmoney.com"],["half day"],["th contract #"],["global database"],["thoughts"],["attached credit sheet"],["one page"],["aec"],["david e deutsch"],["response"],["pap"],["gerald"],["contract price"],["julie"],["mid month"],["debra"],["documents"],["palo alto"],["msa"],["few tweaks"],["revised devon"],["clean documents"],["one more thing"],["tentative conference call"],["economics sheet"],["some"],["revised net"],["huber transaction agreement"],["term"],["attached revision"],["this draft"],["final look"],["attached confirm"],["enron lng"],["link"],["sarah pyles"],["jennifer fraser's memo"],["dogs"],["termination notice"],["ed baughman"],["prudency c summaries"],["6 analysts"],["this guy"],["map"],["jon"],["cinergy"],["kevin \ni"],["epmi"],["format"],["test"],["i'll"],["david w delainey"],["eob"],["enron europe"],["mark woolfolk\n\n"],["nancy pickover"],["weight"],["spreadsheet"],["darrell"],["bp's meeting"],["question"],["ambrosia"],["this document"],["both the el paso"],["obsession"],["incident"],["ted"],["one"],["new isda"],["kim ward"],["cheryl nelson"],["tana's list"],["veronica valdez"],["melissa murphy"],["omnibus"],["tracy ngo"],["credit"],["marie"],["help"],["david vitrella"],["signature block"],["voice mail"],["susan f"],["this e-mail message"],["5 pm flight"],["steve morus"],["guaranty"],["russell diamond"],["boilerplate isda master agreement"],["who jane wilhite"],["mg ltd"],["finance"],["rodney\u002ffinley\u002ft"],["us"],["marcie youngerman"],["julian poole"],["master\" section"],["ect"],["lotus notes"],["tiger"],["counterparty"],["ena signature"],["any questions"],["what lawyer"],["east and west power desks"],["daily termination list"],["anyone"],["mahonia limited file"],["confirm desk"],["master log"],["embedded image"],["mcmahon"],["mike"],["argentine crisis"],["specter"],["filings"],["skilling"],["this interview"],["transmission deals"],["confirm letters"],["leg"],["two smurfit deals"],["puget sound"],["high level"],["assistance"],["billing"],["going"],["any brokers"],["new markets\u002fproducts"],["enron"],["koch"],["few minutes"],["these forms"],["any master agreements"],["mark h"],["sara"],["andrea cal"],["phone call"],["this opinion"],["view"],["end"],["following note"],["any concerns"],["transcolorado"],["independent"],["some names"],["xmas cards"],["gag gift exchange"],["this sheet"],["temperature model"],["kevin lambert"],["ozzie"],["discussions"],["lavo"],["meeting plans"],["gift certificate"],["credit worksheet"],["good place - bay area"],["check"],["lunch"],["ranelle"],["attached interconnect agreement"],["bill"],["lindy"],["man"],["jay"],["lisa - \ni"],["emails"],["reuters"],["louise\n\n\n\nespeed"],["75%"],["apr"],["errol"],["errol pleae"],["buddy"]],"hovertemplate":"Power Score=%{x}\u003cbr\u003eGender Score=%{y}\u003cbr\u003ename=%{customdata[0]}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"","orientation":"v","showlegend":false,"x":[0.05986475735879077,-1.0,0.23529411764705882,0.0784313725490196,-1.0,0.5714285714285714,-0.05806451612903226,0.0625,-0.3333333333333333,0.1546572934973638,0.0,-0.13333333333333333,-0.1744186046511628,0.03787878787878788,1.0,0.4117647058823529,0.0,-0.5,1.0,-1.0,0.1266607617360496,-0.5128205128205128,1.0,0.8,-1.0,-0.14754098360655737,-1.0,0.0,0.06666666666666667,0.0,-1.0,-1.0,1.0,-0.5087719298245614,-0.13333333333333333,-0.18181818181818182,0.0,0.10599078341013825,0.020833333333333332,-0.0726643598615917,0.04032258064516129,0.0,-1.0,-1.0,0.041666666666666664,-0.04,0.0,0.056338028169014086,-0.2542372881355932,-1.0,0.5,-1.0,-1.0,1.0,1.0,-1.0,-0.03854875283446712,-1.0,0.0,-0.06666666666666667,0.3076923076923077,0.0975609756097561,0.09523809523809523,0.0,-1.0,0.14285714285714285,1.0,-1.0,0.07142857142857142,0.0,-0.5,1.0,1.0,-1.0,-1.0,0.0,0.0,-1.0,0.16,-0.3125,1.0,-1.0,0.0,-0.10224438902743142,1.0,-1.0,-0.48148148148148145,0.14285714285714285,-0.5,0.006430868167202572,-0.016129032258064516,-0.17647058823529413,-0.058823529411764705,0.0,0.0,1.0,0.0,0.011904761904761904,-0.14285714285714285,0.0,0.375,-1.0,0.0,-0.6,0.02631578947368421,-0.6666666666666666,-0.4,1.0,0.16666666666666666,-0.011235955056179775,-0.14285714285714285,0.0,-1.0,-0.013333333333333334,-1.0,1.0,-1.0,0.09090909090909091,-1.0,-0.022388059701492536,-1.0,-0.4074074074074074,-1.0,-0.4,1.0,1.0,-0.35714285714285715,-1.0,0.0,0.0,-1.0,0.007987220447284345,1.0,0.6666666666666666,0.0,-0.42105263157894735,-1.0,-1.0,-1.0,1.0,0.0,-0.25,1.0,-1.0,0.3333333333333333,-1.0,1.0,-0.6,-0.09090909090909091,0.3333333333333333,0.16666666666666666,0.15384615384615385,0.3333333333333333,-0.037383177570093455,-1.0,-0.5,0.18421052631578946,-1.0,0.0,-1.0,0.0,0.0,0.8571428571428571,1.0,-0.45454545454545453,0.0,0.0,1.0,-1.0,-0.017391304347826087,1.0,0.0,1.0,-0.4,0.17857142857142858,0.21428571428571427,1.0,1.0,-0.08695652173913043,-0.007936507936507936,0.5833333333333334,-1.0,0.0,-0.1111111111111111,1.0,1.0,0.0,-1.0,-0.5,1.0,-1.0,-1.0,0.08743169398907104,0.10526315789473684,0.0,0.07142857142857142,-0.75,1.0,0.0,1.0,0.0,-1.0,-1.0,-1.0,0.0,0.5,-1.0,0.0,0.2857142857142857,0.0,0.5,0.25,-1.0,-0.12903225806451613,0.0,1.0,0.11764705882352941,0.05263157894736842,-1.0,0.4,1.0,-0.027777777777777776,-1.0,-1.0,1.0,-0.5,-0.014084507042253521,0.25,0.0,0.09090909090909091,0.0,-1.0,-1.0,0.0,0.0,1.0,-0.013333333333333334,-1.0,0.0,1.0,-0.5,0.0,-1.0,0.0,0.4,0.0,0.0,-0.4666666666666667,0.045454545454545456,0.1,0.0,-0.4444444444444444,0.016666666666666666,-1.0,0.011235955056179775,1.0,0.0,-1.0,-1.0,0.0,0.0,0.0,1.0,-1.0,-0.0625,0.6363636363636364,0.0,-1.0,-1.0,-0.19230769230769232,0.0,0.0,0.2857142857142857,1.0,-0.6666666666666666,1.0,-0.16666666666666666,0.3333333333333333,-0.05263157894736842,-1.0,-0.4166666666666667,-1.0,0.0,-1.0,-1.0,1.0,-0.14285714285714285,1.0,1.0,-1.0,-1.0,-0.38461538461538464,-1.0,-0.5,-1.0,-1.0,-0.3333333333333333,0.0,0.0,-1.0,0.1,1.0,1.0,-1.0,0.0,0.0,-1.0,0.2857142857142857,0.0,1.0,-1.0,-1.0,-0.07692307692307693,0.0,-0.05660377358490566,-0.17647058823529413,-0.6,0.0,1.0,-1.0,1.0,0.5,-0.125,1.0,0.0,0.0,-1.0,-0.3333333333333333,-1.0,0.0,0.0,-1.0,0.0,0.0,0.0,-0.3333333333333333,0.0,-1.0,1.0,0.0,0.0,-1.0,0.5,-0.5,0.6,-1.0,0.0,-1.0,0.3333333333333333,-0.2222222222222222,0.0,0.0,0.0,0.0,1.0,0.0,-0.15,-1.0,-1.0,1.0,0.0,1.0,-1.0,-0.3333333333333333,-0.14285714285714285,-0.25,0.0,-0.2222222222222222,-0.05714285714285714,-1.0,0.0,0.0,0.0,-0.14285714285714285,1.0,0.0,-1.0,0.375,0.0,-0.5,0.25,0.0,0.0,-1.0,-1.0,1.0,-1.0,-1.0,-1.0,-1.0,-1.0,0.0,-0.25,1.0,-0.08695652173913043,1.0,0.05263157894736842,-1.0,-0.18181818181818182,0.08695652173913043,-1.0,0.0,0.0,-1.0,-0.4,-1.0,0.0,-1.0,-1.0,0.08333333333333333,-0.5,1.0,0.3333333333333333,1.0,-0.14285714285714285,-1.0,-0.6666666666666666,0.045454545454545456,-1.0,-0.5,-1.0,-0.5,1.0,-1.0,-1.0,-1.0,1.0,0.6666666666666666,1.0,-0.75,-1.0,-1.0,-1.0,-1.0,-0.043478260869565216,-1.0,1.0,0.0,1.0,0.007194244604316547,1.0,-1.0,0.0,0.0,-1.0,0.7142857142857143,0.0,1.0,0.0,0.0,1.0,-1.0,-0.1111111111111111,-0.3333333333333333,-1.0,-1.0,-0.029411764705882353,-1.0,-1.0,1.0,0.0,-1.0,0.1111111111111111,0.0,-1.0,-1.0,0.0,-0.8,0.16666666666666666,-1.0,-1.0,1.0,-0.6666666666666666,0.5,-1.0,0.16666666666666666,1.0,0.0,-1.0,0.3333333333333333,-1.0,-1.0,-1.0,1.0,-1.0,0.0,0.0,-0.5,0.0,0.2857142857142857,0.0,0.0,1.0,-1.0,-0.3333333333333333,0.5,-1.0,-1.0,-1.0,-1.0,1.0,-0.23076923076923078,-1.0,-0.014925373134328358,-1.0,0.0,0.0,1.0,0.0,-1.0,-0.42857142857142855,1.0,1.0,0.0,-1.0,1.0,-0.25,-1.0,-0.16666666666666666,0.08108108108108109,-0.1,-1.0,1.0,-1.0,0.75,-1.0,1.0,0.0,-1.0,-1.0,0.003978779840848806,0.0,0.0,0.0,0.2857142857142857,-1.0,0.0,-1.0,-1.0,1.0,-1.0,-1.0,-1.0,-1.0,0.2,-1.0,1.0,-1.0,0.5,1.0,-1.0,1.0,-1.0,1.0,0.0,0.0,1.0,-0.5,-0.005714285714285714,-0.5,1.0,-0.17647058823529413,-0.0967741935483871,-0.5,0.047619047619047616,-1.0,0.0,1.0,0.0,-1.0,0.3333333333333333,-1.0,-1.0,-1.0,-1.0,-1.0,0.0,-1.0,-1.0,-0.6666666666666666,-1.0,1.0,0.0,0.0,0.0,0.0,0.16071428571428573,-1.0,0.6666666666666666,0.0,0.038461538461538464,-0.5,-1.0,1.0,-1.0,0.0,-1.0,-1.0,-1.0,0.0,-0.35714285714285715,-1.0,-1.0,-1.0,0.0,-0.5,0.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,0.16666666666666666,-1.0,1.0,0.0,0.0,1.0,0.25,0.25,1.0,0.08,-1.0,0.2,0.5,0.0,-1.0,-0.2,-0.16666666666666666,1.0,0.006711409395973154,1.0,1.0,-1.0,-1.0,1.0,-0.6666666666666666,1.0,-1.0,0.16666666666666666,0.2,0.0,-1.0,-1.0,0.0,0.15384615384615385,-0.5454545454545454,0.0,0.021052631578947368,-1.0,-1.0,-0.6923076923076923,0.0,0.7333333333333333,-1.0,-1.0,0.0,0.0,-1.0,-1.0,1.0,1.0,-1.0,0.25,-1.0,0.5,-1.0,0.5,0.2,-0.4,0.25,1.0,-1.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0,1.0,-0.13043478260869565,1.0,-1.0,0.0,0.5,-1.0,0.0,-1.0,-1.0,-1.0,0.0,1.0,-1.0,-1.0,1.0,-1.0,1.0,0.0,1.0,0.0,-1.0,-1.0,-0.5,0.020833333333333332,0.0,1.0,0.0,0.4,0.0,0.0,-1.0,-1.0,1.0,-1.0,-1.0,-1.0,1.0,-1.0,0.0,0.0,0.3333333333333333,0.07692307692307693,1.0,-1.0,-0.14285714285714285,0.0,-1.0,0.1111111111111111,0.0,0.0,-0.04,-0.02127659574468085,0.25,0.2,0.0,-0.5,1.0,1.0,-1.0,-1.0,-0.5,1.0,1.0],"xaxis":"x","y":[0.5201260232790609,0.9335179305299848,0.9825503609888103,0.9874200784217547,0.9383831410763586,0.9841564410135782,0.9831506312864673,-0.5198352613199437,0.6186401228078531,0.9068643254926438,-0.5199737070860624,0.8440421196030113,0.8357781081726421,0.5203662494921178,0.9033911404467658,0.5477693925330205,-0.9871114919912899,-0.5968152296495193,0.8883462916750575,-0.5458813367330698,0.8211006900460458,0.9841673442250817,0.9412536957250028,0.9897285947435532,0.7363307462730648,0.5201369017191932,0.5193166540591116,0.9067532335869544,0.8616094694896165,0.6676164755672235,0.8911364323580012,-0.5196913262735969,0.9876896198116487,0.9398191824223195,0.6688509768665913,-0.8441733897522159,0.9845229690748324,-0.5390445929183305,0.6215253356144336,0.5317343316312051,0.8084885092181746,0.6902306998770068,0.5181517710923673,0.9863807972134683,-0.9840196737137423,-0.8112095737423867,0.5184362134602815,0.7167779572011377,0.8422691882933888,0.8512695133123876,0.9043840518940582,0.5215269078267142,0.8533695242648583,0.946227626933663,0.7197647379721438,0.8432328867336751,0.9089619981726381,0.8718580699142778,0.9835259414733403,-0.5406748683302628,0.9046326546814244,0.8801863762829277,0.8709061514597793,0.9413937143642659,0.861700384485858,0.9840488202510651,0.68837079310747,0.8426818059331986,0.992310187549412,0.9154083004052536,0.8584033351138323,-0.9841210714304501,0.987708098777835,0.9999999998849449,0.7504896888998501,0.9835971149104193,0.9776532516067235,0.9757594073798238,0.8048259334118344,-0.5177847744318139,-0.542228026993688,-0.9633540489679333,0.9976541697181897,0.7432102462184088,0.8484660376619129,0.9392702418069322,0.700951643568602,-0.865390570754324,-0.7304751770159951,-0.6141256734049133,-0.9832452148258919,-0.9840765048508062,-0.8198597562114837,-0.8691509571516584,0.92214684127902,0.8684265222825508,0.814629046590321,0.8007699976452254,0.7953283679157558,0.9839074423945797,-0.814222760707656,0.5201649467064406,-0.6340081256079229,-0.5558170716166221,0.9071924812613827,0.9842003806584759,0.9823527608372913,0.9444332183333165,-0.5590192281781136,0.9926376758861942,0.941722488426363,-0.5397018641356583,0.9175165295576367,-0.7308622455845113,0.7624049216549333,0.8423736568980965,0.9826781470778349,-0.5667921306361817,0.7032646403421339,0.88800833100305,-0.5656376331859261,0.7991267352724293,0.7205997562378089,0.8458561326766921,0.8322920686741813,0.9051551843702584,0.9436693563133223,-0.7281917522795518,0.891429275349705,0.5179518778450145,0.904530249173142,0.614216395733916,0.62453166927098,0.721733094745413,0.8712416444985159,0.7137730103295986,-0.5245014941761261,0.6738932037901845,0.7121852132696901,0.6775199532951968,0.6603405782616106,-0.80930960904369,0.5178506506166579,0.9952722016775634,-0.5175865419427019,0.9524060584535692,0.5265893086176185,0.6974390207752438,-0.9924926407600863,0.8096767061699579,0.7272383113402772,0.745075283038295,0.9905329315256384,0.86764046136182,0.8347815623991781,0.6235509558226099,0.9755659544846109,0.5197439750412931,0.5679943807587502,0.5288725676437592,0.9828790476047312,0.6972264729006911,0.9693698120051987,0.6372183709316666,-0.6642068680600476,0.639744482975116,0.7287321514458076,-0.5384126377871765,0.8939795538780625,-0.9888170665564601,-0.9922387691273362,0.8313639466780555,0.5623457669645233,0.6780836374334799,-0.9836237151410104,0.5680090704055508,-0.813895294460129,0.9026534083209538,0.9338528697778188,0.9879378771177514,0.9825581706576885,0.8912631309868814,0.9042659629914503,0.996363611343378,0.9630993971999149,0.9326823393400833,0.6893771007150901,0.949263485774918,0.5201413690393821,0.9840659487471912,0.8531675241179068,0.9891961797990821,0.991420183596705,0.8530224106422549,0.5469003870336767,0.6951416234744028,0.5188270355366496,0.6609479001715259,0.9414109391724161,-0.9039494800283435,0.5917273888270933,0.5570672511731506,0.787320729907913,0.8495170018392683,-0.519138362761669,0.987699138782132,0.9817514577003605,0.9771678177661349,0.7388874615498999,0.9796136607244218,0.9828484304413544,0.9062434095127943,0.7393037740765869,0.9838577532256942,0.8300594004767808,0.5280118876317589,0.8418425095547101,0.9917590639796819,0.9779029266340312,0.9831692696601162,0.7298621342277085,0.9838034028561284,0.7169625137833986,-0.9842132057619761,-0.9818285205110052,0.8492007662880462,0.7315922364019852,0.9798856850493083,-0.5319076066137698,-0.8946875737476796,0.6584790865118141,-0.992448742475584,-0.541753198418196,-0.9842330329104348,0.5831996239452532,0.9823697435860819,0.9684817210179759,0.5803181912390325,0.5181644237742639,0.7122902200665637,0.8481250709398357,0.9853582171595049,0.7324513701978025,0.8643137348585522,0.9017783116932782,0.9440388633977757,-0.5952801915657926,0.5979232018848077,0.8156464846790578,0.5818665964888456,0.8241989369445095,0.8738432782931835,-0.9649610244136071,0.8955044696996656,0.9833475348134011,0.6898148081570399,0.9144361472265932,-0.564649155544314,0.9399661601277672,0.8433841261866316,0.9016655781153169,0.8854530501456536,0.7276881194978934,0.9417548456746397,0.9832382503231123,0.5803181912390325,-0.9921197312221824,0.6935643517024077,0.9999972921349148,0.9802300811294672,0.6718461143260869,0.6202974664571888,-0.5180955869719682,0.5821916383587193,0.9248852332690303,0.8522187733580947,0.8065800218279028,0.8908345268911034,0.7242158386883282,0.7098889565857104,-0.645150921079618,0.8241111827707888,-0.5573527402992948,-0.8624651032835954,-0.6387475848917411,-0.9842503281651824,0.7977695104384236,0.9835653320609664,0.8260577817293531,0.973122685879592,-0.889046297961255,0.9910108136941175,0.9819678533396005,-0.7063011054876869,-0.8654571874521796,0.9997260529363262,0.5683040745331682,-0.520204410313897,0.5206756920823042,0.8332629597855817,0.8538412507451411,0.878394593510576,0.9792277529118566,0.668095346494564,0.8426576675102593,0.5204653609123941,0.9298642563068015,-0.9814618510997192,0.520270515110603,-0.9344816899542185,0.5266602796087035,0.9986323701370765,0.9965204973822427,0.997219917860702,0.5180821904297181,0.8220982545147265,0.988307701827754,0.8823819275424273,0.9838554275558744,0.7623574032198007,0.9839150116292512,-0.638963981799445,0.7120178382539524,0.6626083302128115,-0.9857391368939321,0.6846185085347595,0.8892876780324926,0.558878421244566,0.688344865965421,0.9836026102030149,0.5175938366811598,0.8887390996667908,0.6628453133449481,0.5736075480164637,0.9135518710804472,0.7044634124816073,0.536779821198204,0.8600449096469498,0.6888362588294128,0.9824725550040135,0.9574592154487105,0.7219208154439102,-0.9930396465115126,0.6843830559233474,0.5176691613413011,0.9823608655859066,0.9974653069695285,0.9999999985879204,0.9841770841851036,0.9555641269571854,-0.9772669187154571,0.9925777165243322,-0.5183782753167655,0.9806580726430948,0.9864332233193963,0.9842242163389625,0.9827188429058646,0.980820790107848,0.6794556848068003,0.9823972575216948,-0.6807132526848346,0.5325298810126629,0.5408650499519115,0.6819586347953683,-0.5206935072298856,0.6088138835370859,-0.7669043552145972,0.7076387098727382,0.888444375767983,0.9551742025237655,0.7286694227237681,0.9791384031362205,0.9823432192396115,0.9840711717921714,-0.8837429269868669,0.7007270087068821,0.9999510820687993,0.9934713473918904,0.9343264815882372,0.8846764270354105,0.6688635617736223,0.9842433543009717,-0.5176359089265804,-0.9834400705136742,-0.8883002331513705,0.9741469869439517,0.656484665406988,0.9822562450311313,0.7309307495561095,0.5446117452314545,0.6880439218335274,0.7958812616819859,-0.9942834556540444,0.9680871207904084,0.7657031427868197,-0.5734121910240069,-0.8496772870162185,0.7204839397496345,0.6688123426667661,0.9825292510579277,0.9564853701977745,-0.7191644481684143,0.6930146720659858,-0.5723692267473074,-0.525821491762303,0.9837255969593534,0.9822643919949262,0.9664936846303133,0.9826887881535691,0.6548635064177303,-0.944140248017027,0.9054338740718296,0.9842298373320287,0.6706837106252935,0.9842171148249002,0.9312258567746172,-0.7124903816842608,-0.942594860773671,0.7194265309613985,0.8627661139863101,0.936539702648379,0.7058561029219547,0.962322145906313,0.7267043842070557,0.5881062322362123,0.9836949763714168,0.6507189920509685,0.7814865719084071,0.9825489013822006,0.966772068491223,0.5620399643219038,0.9416121639767248,0.9837355386118687,0.8365168827617415,0.7839402147572656,0.9836435143264017,-0.5204480877929696,0.5862898894286754,-0.9803811937907099,-0.9521391200348895,0.9795261015584865,0.5673736733018199,0.988340365269342,0.9840277578174573,0.5840241646763714,0.9459366817036801,0.8433443090542098,-0.8415834569983989,0.767894110012512,-0.9739802479548467,0.789982710729971,-0.520217815697637,0.983025947849962,0.5578465153696887,0.9150065932868288,0.9157094340291445,0.9729614269117667,0.9999866263170177,0.9839248467787642,0.7127852168400101,0.9084354674567715,0.9129768806394323,0.6625859762216332,-0.947828156045405,0.5976486150647682,-0.5677327345075867,0.979732072118467,0.7362522646464786,0.9915096120109834,0.979212554360269,0.9798090141273561,-0.5852672525472833,0.9952366896521304,0.6770954081129306,0.6550609736448874,0.9829647157101636,0.9929218418133146,0.9911104260037431,0.9089516304273023,0.973651510975462,0.891607513290445,0.8462788545577897,0.9999957913928926,0.6446946424025023,0.999998846246813,-0.5741664221993413,0.7087142521894714,0.9833475348134011,0.5954684611801392,0.9917590639796819,0.5297595618972656,0.6407865472387431,0.9726698502067989,0.9354249679122281,0.6245298723445202,-0.637931983166462,0.9448837883636018,0.7928071486901518,0.7316465084871473,0.9999940094495791,0.8502516385944315,0.9288404093425141,0.974501692794811,0.8414051139051142,0.8510655572419479,0.9922734158225701,0.5540588114227781,0.9814654933982946,0.6900484620532354,0.7188211543953369,0.5846955806049835,0.9829158705684294,0.5224290447511707,0.5361942233670279,0.9843772483151749,0.901319519712383,0.8183913289546126,0.8339000274104285,0.9911805450420837,-0.9744827668904703,-0.9739802479548467,0.7219503600035508,0.6487218756892699,0.5356513483219363,0.6751526825684898,0.9537229903164237,0.6693997421921019,0.8710182972618201,0.8692911612792762,0.6876785613422307,0.621480772129348,0.918773869658441,0.9700969204265366,0.8846080045699194,0.8583692201800153,0.6780390441463834,-0.6083889665301155,0.5975272865373334,0.9823710322751485,0.7232792744711707,0.9840145577178266,0.982905916601028,0.9933482207641123,0.9996659316607938,0.7082451946184223,0.715920676580323,0.9756462529991111,0.8483569230094412,0.86764046136182,0.9060726186089915,-0.662362523002717,0.5337549362497044,0.7061288316029843,0.9255595525060591,0.8100326597777456,-0.6086476328371796,-0.5261045700799284,0.9812469745458124,0.6873756082987049,-0.5184653305134868,0.9827996693458307,0.9833147888132129,0.5203095458444463,0.5551665432591624,0.8743167409567314,-0.6619181768455662,0.6486550250561453,0.5359724358699961,0.6395444293014481,0.9821753789373282,0.8152677932024707,0.9980748967603781,0.9802147707936905,0.7174483569924356,0.9697407753902555,0.7033221835845692,0.6678904377430537,0.6633905247086531,0.8291790178051888,0.5653321689849758,0.9871748222470472,0.6887461469873236,0.7231673271475924,0.9821888853456088,-0.5199174138945909,0.91772815566381,0.9915125362383843,0.8575980358531287,-0.9837059734264715,0.7925802693650508,-0.9882040909981591,0.5917791700237091,0.6787348802462111,0.6491841589125084,0.9840435836148338,0.9810206421973283,0.8831974587248754,0.5676151781135889,0.6369638702071587,0.8601401228865765,0.7824935455290751,0.9120260830047365,0.9836623314825957,0.8545570296390106,0.5594896212607627,0.9479432090625538,0.9613465075889883,0.8178836261907063,-0.7144565863029593,0.9767057561053872,-0.8366598416747699,-0.6871548041874049,0.7276883578094637,0.6363798845303882,0.9860458333905878,-0.933046000738264,0.8100681025216455,0.9827789196195426,0.6694622355580373,0.9948781287773659,0.5383838522371605,0.9761343726141991,0.5226844844076349,0.6997644837812445,0.8618241209320087,0.8317374173062348,0.9834554617411315,0.7210516367659409,0.856247472551896,0.9830526460303447,-0.9037995496388301,0.6062945048515891,0.6569138203215217,0.9445360973044071,0.855441256042797,0.6823147200120934,-0.9927299362613003,0.8847942688271607,0.7729818187581796,0.7294785677921488,-0.545850822619335,0.9951442245911933,0.6978524087678835,0.7371564586938782,-0.6375780640134583,-0.7956832203572949,0.8401977712553151,-0.9822413462211669,-0.9829183582608945,0.6879505994301148,-0.6996948928942065,0.5798098179778864,-0.9842072949948792,0.6503269629968561,0.9166645160564493,0.7133905137121763,0.8198683365021557,-0.806625185179635,0.9596397013756984,0.9821968969782452,0.9626435883854009,-0.5421886145230955,0.9779525796778952,0.565594024504496,-0.9249097893345264,0.7288138818204988,-0.5423731725846037,0.6245888664741693,0.901316566408285,-0.9415017311578936,0.9819814617708748,0.6467369766504838,0.5883253995599681,-0.6261490555179614,0.9197119620278943,-0.5204128759008587,-0.9839590059422735,-0.9431874899071537,0.9840877055244355,0.9146652201660064,-0.7270555888512191,0.6372981851165486,0.6322994533690418,0.662742082220673,0.9695028631203555,0.8467420596408204,-0.6635364152627691,0.8652845828096418,-0.9339209856964126,0.6961750038555257,0.6958404110192433,-0.5309690792726346,0.9437921022393975,0.9013055220014168,0.865859923158052,0.5682021539836768,0.8505244722064399,0.9827109810151506,0.6678148055519414,-0.5363277952323016,-0.5187743892089617,0.7231179357189258,-0.8642476051412107,0.8529390581868961,0.6349907634583783,0.7291806106411213,0.9841481821491626,0.7341002883415756,-0.9183175886967806,0.9820422644856872,-0.9776977184154566,-0.728816596153345,0.7240378432366996,0.9789390373812568,0.8855201448529153,0.8148719384040302,0.6658806912022881,-0.9378159976989966,0.6928549266952503,0.8913311962326074,0.8465899890795798,0.7405462791393136,0.6211976656870916,0.9419883119790777,0.8772143769579324,0.9761779244023268,0.9350351826344706,0.5868605812488212,-0.7035492224236701,0.7313092528814362,-0.520240089478379,0.6274345010073548,0.9819525296266735,0.98818771747446,0.691689132317376,-0.9878093177687092,0.6597567387355993,0.9854836004713581,-0.8882337324987016,0.8818946184577006,0.911465658309473,-0.980366191680252,0.550265746729542,0.6752841080410088,-0.5903568242706572,0.9839526716939377,0.8742889580955235,0.9820551408803551,0.9824825766458765,0.9838182632410204],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Power Score"},"zeroline":true},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Gender Score"},"zeroline":true},"legend":{"tracegroupgap":0},"title":{"text":"Power vs Gender Scores"},"shapes":[{"line":{"color":"black","width":1},"type":"line","x0":0,"x1":1,"xref":"x domain","y0":0,"y1":0,"yref":"y"},{"line":{"color":"black","width":1},"type":"line","x0":0,"x1":0,"xref":"x","y0":0,"y1":1,"yref":"y domain"}]},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('b9df4588-235c-4886-a5d1-859813c1eb2e');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



    
![png](clustering-on-gender-and-power_files/clustering-on-gender-and-power_48_1.png)
    


As a supplement to the boxplot and scatterplot results I made histograms showing gender and power distribution. This below code outputs a single dimensional view for both power and gender score.


```python
#histograms of each axis
fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

#power
axes[0].hist(gender_power_df['power_score'], bins=20, edgecolor='black')
axes[0].set_title('Power Score Distribution')
axes[0].set_xlabel('Power Score')
axes[0].set_ylabel('Frequency')

#gender
axes[1].hist(gender_power_df['gender_score'], bins=20, edgecolor='black')
axes[1].set_title('Gender Score Distribution')
axes[1].set_xlabel('Gender Score')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```


    
![png](clustering-on-gender-and-power_files/clustering-on-gender-and-power_50_0.png)
    


### Riveter Analysis Results -> Interesting, but lets get more data
- Before looking into the results, its worth noting that I pulled an ordered subset of the data. I didn't want to randomly sample as that would reduce related verb subject counts pretty significantly, but this does somewhat reduce the power of my conclusions to what names nltk can correctly identify. Also, when looking at the distribution of gender, the sample size for female labellings is much smaller than that for the male labellings. This isn't surprising given the barchart from earlier showing the sender-receiver gender distributions.
- Recall from the research hypothesis: "Men will show more power-associated language than women,". This data does not support the claim, given that the spread of the female names have a higher power ranking than the male names at all gender confidence levels. Considering name-verb pairs as involvment in power through text, rather than the direct application of power, these results could support the idea that in a male dominated gender environment, women feel less inclined to participate, which could be expressed through mapping verbs to objects more often than to names. Prabhakaran's paper also expressed a much higher discrepency in the mean ODP (overt displays of power) counts of male superiors than any other group. Additionally, female-subordiante ODP was the next highest, followed by male subordiante, followed by female superiors (Prabhakaran et. al). Steffensmeier's study concluded that Enron didn't have many "highly placed or powerful white-collar female criminals." What this could entail is the idea of "relational closeness" and groups within conspiracy being primirily male dominated, and that these clusters were the objects where ODP's were expressed at Enron.
- While these results do support this sort of story, looking at the scatterplot, again, shows inconsistency in the gender classification. The API has trouble differentiating words from uncommon female names in America, as can be seen with the word 'he' (chinese female name) at ~ (0.33, -0.5) which's position might otherwise show that males do on average exhibit than accept power in the dataset. However, 'she' does exhibit a higher power score of ~ (0.375, -0.8), supporting the boxplot results. The two causes for indecision with these results are 1, the incompleteness of the emails profiled, and also 2, the genderizing API that I used. The high-confidence results are still majority real people being considered, but inconsistencies in labelling a basic male word like 'he' puts the object gender spread into question as relates to a US-based company like Enron. These issues support continued implemenation of an llm genderizer that can better distinguish between objects, gendered pronouns, and names in-context as directed through prompt engineering.

#Part 4: Tokenization Approach

The previous section went into the aggregate results of gender and power, addressing points 1 and 2 of the hypothesis analysis, but while it supported an alternative idea of power-laden male-dominated subnetworks it didn't give any indication of proof or shape. This next section seeks to use the cleaned data (using nltk for gender and names from the sql merged dataset) in tandem with the llm scraped messages to tokenize the messages and analyze how position and gender factor into any potential high volume clusterings of powered messages, as theorized in thre previous section.


```python
#This pulls the llm-cleaned messages with the tags from part 1
data = pd.read_csv("/content/gdrive/MyDrive/colab-output/riveter-test/riveter-nlp/riveter/22000_llm_df_mini.csv")
token_df = pd.DataFrame(data)
token_df.head()
```





  <div id="df-2630c25c-976f-48ab-a290-57c26ca002e0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>message</th>
      <th>sender_email</th>
      <th>receiver_email</th>
      <th>sender_name</th>
      <th>sender_title</th>
      <th>receiver_name</th>
      <th>receiver_title</th>
      <th>sender_first_name</th>
      <th>receiver_first_name</th>
      <th>sender_last_name</th>
      <th>receiver_last_name</th>
      <th>sender_gender</th>
      <th>receiver_gender</th>
      <th>id</th>
      <th>id_gendered</th>
      <th>message_truncated</th>
      <th>message_llm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>allen-p/_sent_mail/10.</td>
      <td>Message-ID: &lt;15464986.1075855378456.JavaMail.e...</td>
      <td>phillip.allen@enron.com</td>
      <td>john.lavorato@enron.com</td>
      <td>Philip Allen</td>
      <td>NaN</td>
      <td>John Lavorato</td>
      <td>CEO, Enron America</td>
      <td>Philip</td>
      <td>John</td>
      <td>Allen</td>
      <td>Lavorato</td>
      <td>male</td>
      <td>male</td>
      <td>2</td>
      <td>1</td>
      <td>Traveling to have a business meeting takes the...</td>
      <td>I would suggest holding the business plan meet...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>allen-p/_sent_mail/1000.</td>
      <td>Message-ID: &lt;13505866.1075863688222.JavaMail.e...</td>
      <td>phillip.allen@enron.com</td>
      <td>randall.gay@enron.com</td>
      <td>Philip Allen</td>
      <td>NaN</td>
      <td>Randall Gay</td>
      <td>NaN</td>
      <td>Philip</td>
      <td>Randall</td>
      <td>Allen</td>
      <td>Gay</td>
      <td>male</td>
      <td>male</td>
      <td>4</td>
      <td>2</td>
      <td>Randy,\n\n Can you send me a schedule of the s...</td>
      <td>Phillip,\n\nI have been</td>
    </tr>
    <tr>
      <th>2</th>
      <td>allen-p/_sent_mail/107.</td>
      <td>Message-ID: &lt;2465689.1075855687605.JavaMail.ev...</td>
      <td>phillip.allen@enron.com</td>
      <td>david.delainey@enron.com</td>
      <td>Philip Allen</td>
      <td>NaN</td>
      <td>David Delainey</td>
      <td>CEO, Enron North America and Enron Energy Serv...</td>
      <td>Philip</td>
      <td>David</td>
      <td>Allen</td>
      <td>Delainey</td>
      <td>male</td>
      <td>male</td>
      <td>15</td>
      <td>5</td>
      <td>Dave, \n\n Here are the names of the west desk...</td>
      <td>Dave, \n\n Here are the names</td>
    </tr>
    <tr>
      <th>3</th>
      <td>allen-p/_sent_mail/129.</td>
      <td>Message-ID: &lt;21687372.1075855688094.JavaMail.e...</td>
      <td>phillip.allen@enron.com</td>
      <td>john.lavorato@enron.com</td>
      <td>Philip Allen</td>
      <td>NaN</td>
      <td>John Lavorato</td>
      <td>CEO, Enron America</td>
      <td>Philip</td>
      <td>John</td>
      <td>Allen</td>
      <td>Lavorato</td>
      <td>male</td>
      <td>male</td>
      <td>39</td>
      <td>6</td>
      <td>John,\n\n Denver's short rockies position  bey...</td>
      <td>John,\n\n Denver's short rockies</td>
    </tr>
    <tr>
      <th>4</th>
      <td>allen-p/_sent_mail/13.</td>
      <td>Message-ID: &lt;32300323.1075855378519.JavaMail.e...</td>
      <td>phillip.allen@enron.com</td>
      <td>james.steffes@enron.com</td>
      <td>Philip Allen</td>
      <td>NaN</td>
      <td>James Steffes</td>
      <td>Vice President, Government Affairs</td>
      <td>Philip</td>
      <td>James</td>
      <td>Allen</td>
      <td>Steffes</td>
      <td>male</td>
      <td>male</td>
      <td>40</td>
      <td>7</td>
      <td>Jim,\n\nIs there going to be a conference call...</td>
      <td>Jim,\n\nI have been working</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2630c25c-976f-48ab-a290-57c26ca002e0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2630c25c-976f-48ab-a290-57c26ca002e0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2630c25c-976f-48ab-a290-57c26ca002e0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-02cbac4a-c4c6-48d4-86bd-278c1982882d">
  <button class="colab-df-quickchart" onclick="quickchart('df-02cbac4a-c4c6-48d4-86bd-278c1982882d')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-02cbac4a-c4c6-48d4-86bd-278c1982882d button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




When considering a clustering analysis, and given my limited domain knowledge on the Enron hierarchy, DBSCAN seemed like a very appealing option. It figures out the cluster selection itself, and hyperparameter tuning would IN THEORY produce the optimal clustering output for analysis.


```python
#check for nulls in message_llm column and fill
print(len(token_df))
print("Num nulls in 'message_llm':", token_df['message_llm'].isnull().sum())
token_df['message_combined'] = token_df['message_llm'].fillna(token_df['message_truncated'])
messages = token_df['message_combined']

#vectorize
vectorizer = TfidfVectorizer(stop_words='english',
                             max_df=0.9,
                             min_df=2,
                             ngram_range=(1, 2))
X = vectorizer.fit_transform(messages)

#after tuning best parameters were:
best_eps= 0.3
best_min_samples= 3
best_score= 0.6206516526153429

# #hyperparameter tuning for best dbscan clustering
# best_eps = None
# best_min_samples = None
# best_score = -1

# #range of eps and min_samples values to test
# eps_range = np.arange(0.3, 1.0, 0.1)
# min_samples_range = range(3, 10, 2)

# #iterate using sillhouette score to track best parameters
# total_iterations = len(eps_range) * len(min_samples_range)
# with tqdm(total=total_iterations, desc="DBSCAN Tuning") as pbar:
#     for eps in eps_range:
#         for min_samples in min_samples_range:
#             pbar.set_postfix({'eps': eps, 'min_samples': min_samples})
#             dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
#             labels = dbscan.fit_predict(X)
#             if len(set(labels)) <= 1:
#                 pbar.update(1)
#                 continue
#             score = silhouette_score(X, labels, metric='cosine')
#             if score > best_score:
#                 best_score = score
#                 best_eps = eps
#                 best_min_samples = min_samples
#             pbar.update(1)

# print(f"Best eps: {best_eps}, Best min_samples: {best_min_samples}, Best score: {best_score}")

#run with chosen hyperparameters (I ran the above code previously)
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples, metric='cosine')
token_df['cluster'] = dbscan.fit_predict(X)

#running pca on the data for vis purposes
pca = PCA(n_components=2)
with tqdm(total=X.shape[0]) as pbar:
    pca_X = pca.fit_transform(X.toarray())
    pbar.update(X.shape[0])
token_df['pca_x'] = pca_X[:, 0]
token_df['pca_y'] = pca_X[:, 1]

#visualize on plot
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

sns.scatterplot(
    x='pca_x', y='pca_y',
    hue='sender_gender',  #can pick other metrics here
    style='cluster',
    palette='Set1',
    data=token_df,
    s=100
)
plt.title("DBSCAN Clusters Colored by Sender Gender")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()
```

    22119
    Num nulls in 'message_llm': 1162


    100%|██████████| 22119/22119 [00:49<00:00, 446.28it/s]



    
![png](clustering-on-gender-and-power_files/clustering-on-gender-and-power_56_2.png)
    



```python
#output got wacky with how many clusters there were so here is a truncated version
img = Image.open('dbscan_output_truncated.png')
plt.imshow(img)
plt.axis('off')
plt.show()
```


    
![png](clustering-on-gender-and-power_files/clustering-on-gender-and-power_57_0.png)
    


As seen above, the DBscan output had way too many clusters, and didn't leave much room for interpretability or being able to ananlyze potential role or gender based clusterings. This 2000 cluster run with optimal hyperparameters could potentially be due to the wide range of subjects and direct objects analyzed in each message. Runs with a higher minimum distance didn't perform much better, generally having around 600 clusters. I hoped that cosine similarity could mimic a similar verb power analysis like in the riveter model, however, the clustering doesn't have the same hard coded weights. However, the main purpose of this analysis was to look for potential clusterings exhibiting powered language, so I decided to use a kmeans approach instead to set a potentially less optimal, but more interpretable cluster number.


```python
"""
Dbscan had way too many clusters even if it was optimal, so I'm going to go
with kmeans to try and get something more interpretable
"""
#run kmeans
n_clusters = 10  #arbitrary cluster number
kmeans = KMeans(n_clusters=n_clusters, random_state=3350, n_init=10)
token_df['cluster'] = kmeans.fit_predict(X)

pca = PCA(n_components=2)
pca_X = pca.fit_transform(X.toarray())
token_df['pca_x'] = pca_X[:, 0]
token_df['pca_y'] = pca_X[:, 1]
```


```python
#kmeans with coloring by sender gender
sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
sns.scatterplot(
    x='pca_x', y='pca_y',
    hue='sender_gender',
    style='cluster',
    palette='Set1',
    data=token_df,
    s=100
)
plt.title("KMeans Clusters Colored by Sender Gender")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.tight_layout()
plt.show()
```


    
![png](clustering-on-gender-and-power_files/clustering-on-gender-and-power_60_0.png)
    


With kmeans, the clusters are much more interpretable, with 5 main clusters, and the remaining only recognizing subtler differences around (0,0). Here, analyzing sender gender, notable gender clusterings include the primarily female dominated cluster 2 from component 2 0.4 to 0.8, the primarily male cluster 4 in the bottom left, and the high male density in cluster 8 around 0.2.


```python
#kmeans with coloring by receiver gender
sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
sns.scatterplot(
    x='pca_x', y='pca_y',
    hue='receiver_gender',
    style='cluster',
    palette='Set1',
    data=token_df,
    s=100
)
plt.title("KMeans Clusters Colored by Receiver Gender")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.tight_layout()
plt.show()
```


    
![png](clustering-on-gender-and-power_files/clustering-on-gender-and-power_62_0.png)
    


When looking at receiver gender, we see some similar results to sender gender, The more homogenized clusters 2 and 4 are significantly more mixed in terms of receiver gender, and the male-dense portion within cluster 8, expands towards lower pca component 1. Cluster 3 also shows higher male density around pca component 1 = 0.45.


```python
#kmeans with coloring by sender title
sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
sns.scatterplot(
    x='pca_x', y='pca_y',
    hue='sender_title',
    style='cluster',
    palette='Set1',
    data=token_df,
    s=100,
    alpha = 0.5
)
plt.title("KMeans Clusters Colored by Sender Title")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
plt.tight_layout()
plt.show()
```

    <ipython-input-44-f7425a5ee4d8>:17: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all axes decorations.
      plt.tight_layout()



    
![png](clustering-on-gender-and-power_files/clustering-on-gender-and-power_64_1.png)
    


This clustering above, and the receiver cluster below show the positions of teh employees displayed in the clustering. On the sender side, most emails by vice presidents are in the clusters at pca component 1 = 0. This is also the case on the receiving side, indicating that vice presidents share a similar lack of pca component 1, but have varied expression on component 2. We can see in the mapping that much of clusters 2 and 5 had positionless (given the position mappings from the custodians github) employees, which is illustrated by the lack of colored points in these positions. By grouping by position, we also see a decrease in variability, as the unlabelled positions made up most of the noise branching away from pca component 2 = 0 on the right side of the graph.


```python
#kmeans with coloring by receiver title
sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
sns.scatterplot(
    x='pca_x', y='pca_y',
    hue='receiver_title',
    style='cluster',
    palette='Set1',
    data=token_df,
    s=100,
    alpha = 0.5
)
plt.title("KMeans Clusters Colored by Receiver Title")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
plt.tight_layout()
plt.show()
```

    <ipython-input-45-e36a49baba53>:17: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all axes decorations.
      plt.tight_layout()



    
![png](clustering-on-gender-and-power_files/clustering-on-gender-and-power_66_1.png)
    


Given these groupings by power, its difficult to see explicit association of these positions. We can use them as discrete power proxies, however, and view the clusters given what we know about their gender distribution. In the below code, I ranked positions by relative power, which assigned a score to influence the brightness of each point (high power = low brightness). By mapping these encoded power scores with transparency onto the clustering, it allows us to look at concentrations of power within the clusterings by both sender and recipient power.


```python
"""
This is kind of cheating, but I manually listed the positions by
power. I feel like generally finding a secret position of power
wasn't what I was looking to do here, so the only thing that this is
really impacting is reproducability.
"""
token_df['sender_title'].unique()
ordered_titles = [
    "CEO",
    "CEO, Enron North America and Enron Energy Services",
    "CEO, Enron America",
    "President, Enron Global Markets",
    "President, Enron Online",
    "President, Enron Gas Pipeline",
    "Chief Operating Officer",
    "Vice President, Chief Financial Officer and Treasurer",
    "Vice President, Enron Wholesale Services",
    "Vice President, Enron Online",
    "Vice President, Energy Marketing and Trading Florida",
    "Vice President, Government Affairs",
    "Vice President, Regulatory Affairs",
    "Vice President, Vice President & Chief of Staff",
    "Vice President",
    "Managing Director, Legal Department",
    "Managing Director",
    "Director, Pipeline Business",
    "Director",
    "General Counsel",
    "Manager, Chief Risk Management Officer",
    "Manager, Risk Management Head",
    "Manager, Real-Time Trading Desk",
    "Manager, Logistics Manager",
    "Manager",
    "In House Lawyer",
    "Government Relation Executive",
    "Trader",
    "Trading",
    "Analyst Risk Management",
    "Senior Analyst Cash",
    "Senior Specialist",
    "Sr. Specialist",
    "Administrative Assistant",
    "nan"
]
```


```python
#create a rank column based on the power ordering array
title_to_rank = {title: i for i, title in enumerate(ordered_titles)}
token_df['sender_title_rank'] = token_df['sender_title'].map(title_to_rank).fillna(len(ordered_titles))
token_df['receiver_title_rank'] = token_df['receiver_title'].map(title_to_rank).fillna(len(ordered_titles))

#generate a gradient color based on rank
power_palette = sns.light_palette("navy", as_cmap=False, reverse=True, n_colors=len(ordered_titles))
def map_color(title_rank):
    return power_palette[int(title_rank)] if title_rank < len(power_palette) else "lightgray"

token_df['sender_title_color'] = token_df['sender_title_rank'].apply(map_color)
token_df['receiver_title_color'] = token_df['receiver_title_rank'].apply(map_color)

#shapes for clusters
markers = ['o', 's', '^', 'P', '*', 'X', 'D', 'v', '<', '>']
```


```python
#sender title kmeans with greyscale
sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))

for cluster_id in range(n_clusters):
    cluster_data = token_df[token_df['cluster'] == cluster_id]
    plt.scatter(
        cluster_data['pca_x'], cluster_data['pca_y'],
        c=cluster_data['sender_title_color'],
        s=100, edgecolor='none', alpha=0.2, marker=markers[cluster_id % len(markers)],
        label=f'Cluster {cluster_id}'
    )

plt.title("KMeans Clusters Colored by Sender Title Rank with Shapes for Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
plt.tight_layout()
plt.show()

```


    
![png](clustering-on-gender-and-power_files/clustering-on-gender-and-power_70_0.png)
    


Above, looking at sender title on the clusters, we can confirm the previous intuition that the vertical offshoot at pca component 2 = 0.4 and above is dominated by low power/unlabeled individuals. Additionally, we can see concentration of powerful positions in the pca component 2 < 1 range (excluding the cluster of nonlabelled people nearby) and around the male recipient cluster at pca component 1 ~0.4. The large clustering at (0,0) is mixed, but does contain a significant portion of higher position messaging.


```python
#receiver title kmeans with greyscale
sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))

for cluster_id in range(n_clusters):
    cluster_data = token_df[token_df['cluster'] == cluster_id]
    plt.scatter(
        cluster_data['pca_x'], cluster_data['pca_y'],
        c=cluster_data['receiver_title_color'],
        s=100, edgecolor='none', alpha=0.2, marker=markers[cluster_id % len(markers)],
        label=f'Cluster {cluster_id}'
    )

plt.title("KMeans Clusters Colored by Receiver Title Rank with Shapes for Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
plt.tight_layout()
plt.show()
```


    
![png](clustering-on-gender-and-power_files/clustering-on-gender-and-power_72_0.png)
    


The receiver color mapping showcases similar distribution as the sender color mapping. This indicates that pca component positioning is likely more dependent on an aggregated sender-recipient relationship than simply the sender or receiver position demanding a certain type of text input. This while this distinction makes it more difficult to stricly denote high mixed and low 'power environments', given the similar cluster groupings, there generally seem to be power concentrations around (0.4,0), mixed to high around (0,-0.3), and a high volume mixed concentration at (0,0).

### Tokenization Approach Results
- I discussed how one of the primary goals of this tokenization approach was to analyze the research goal of "relational closeness". As shown in the DBscan, the tokenization was likely analyzing disjoint features to the riveter results. The following notable gender kmeans clusterings were located for senders and receivers, with the rest of the data being relatively mixed, reflecting an overarching mixed gender environment at Enron:
  - Sender:
    - cluster 2 from component 2 0.4 to 0.8 -> primarily female
    - cluster 4 in negative component 2 values -> primarily male
    - cluster 3 at component 1 ~0.45 -> primarily male
    - cluster 8 at componeent 1 ~0.2
  - Recipient:
    - clusters 2&4 more mixed than for sender
    - cluster 3 -> primarily male
    - larger cluster 8 male segment

- When looking at the clustered title colorings with these clusters in mind, cluster 8 is the real standout, as the female dominated cluster 2, was largely non-labelled employees. The lack of a distinct female-executive cluster could support the idea of a more 'hands off' female disposition within a male dominated executive gender environment. Cluster 4 is likely not related to power, at least in a traditional sense. The mixed title power of this cluster likely implies that these predominately male emails have some other relational factor. Lastly, cluster 8 gives rise to a potential male-dominated grouping. While the explicit exclusion factor isn't visible, the outcome of segmentation is, when looking at the results. While it is possible that there is only one such cluster, this broader appraoch of attempting to create and map a continuous variable for power, damages the visibility of a single contriubutor.

## 4. Discussion and Conclusions
#### Conclusion
> "Individuals within a workplace network adjust their communication style, specifically their choice of verbs, tone, and formality based on hierarchical position, gender dynamics, and perceived relational closeness. Men will show more power-associated language than women, while senders will align their communication style with personas tied to their workplace role."

- By modelling the workplace network, and caputuring verb use, tone, and formality through riveter, I defined 'power' of an entity within this network, and compared gendered data points on this continuum to show empirically that women in general exhibited higher power in Enron workplace communication.
- Enron's documented criminal activity as discussed by Steffensmeier et. al shows that work channels were used as a platform for non-work criminal networking, and the theory developed from the riveter results that there may be a male-executive cohort with lots of power exhibition, as supported by the claim that there weren't prevalent criminal executive women at enron engaged heavily in these networks. Using kmeans clustering supported by nltk and an llm message body extracter, I modelled message similarity in two dimensions, and did notice a predominately male clustering of messages to and from high power work titles. This could support the idea of relational closeness between the Enron employees, with the larger (0,0) cluster representing standard work activity, and the
- These findings did not explicitly reveal any paradigm for downward communication, given that the sender and recipient clusterings weren't significantly different. This could however support the idea of alligning both ones communication style and the message to that person based on their role, which lines up with conclusions from past research detailing the effect of gender environment on employee messaging.
- One consideration for future study is how important verb choice will be with the advent of proprietary AI systems in companies across the US. I know when I need to write an email I will ask an LLM to help me format. Delegating the power consideration could reduce the usefulness of this task, however, these LLM's are still trained on data like Enron that contains these power embeddings. I think gender-power analysis like this should be applied to social networks to see power imbalance or potential abuse in direct messageing. I also think it could be used to find a leadership core in group discussion, even if one doesn't formally exist.
#### Methodological Effectiveness
- riveter package
  - The set up was a bit tedious, and it required an older version of transformers to run, which made finding a huggingface model with enough parameters that worked on the same version a bit tedious. However, the power analysis was very helpful, and their documentation made it pretty simple to implement beyond that. In terms of the results, the package provided a strong metric of power beyond simply ranking titles that didn't rely on domain expertise. I feel that it was a very good fit for this task, however when working in a more informal setting (ie. instagram or twitter data) how would it deal with non-traditional verbs, sentence structure, and mispelling I'm not sure.
- LLM prompting as a means for data cleaning
  - Huggingface makes it very easy to go about model selection, and apart from certain parameters requiring outside access, and some models requiring additional permissions, the rest of the implementation would be pretty cookie-cutter for additional text mining tasks. The message cleaning function worked well, especially given the low quality model that I was using. As I said before, that task is more cleaning less generating so this behavior makes sense. The one issue I did run into was the size of the context window being too small. this approach could be fixed by batching, even though batching would reduce the connectedness of some text data that might need to be analyzed in other tasks.
- Genderizing message senders and recipients
  - nltk
    - this approach I felt was generally the most reliable. All I had to do was download the names data and the functionality did the rest. This approach was a bit rigid, and combined with regex it wasn't a natural fit to getting the sender and recepient names directly from the message. However, with the sql merge from the custodian dataset, it had a clear list of names to genderize, and it performed much better than the other methods. However, it will likely not be the case in other adjacent projects that data like this already exists.
  - namesor API
    - Overall I felt like this api did a pretty good job. It had a bit of a tough time given that it was presented with both people and objects. However, the confidence score that it gave returned pretty good accuracy at higher confidence. The flexibility of the input is pretty good, and I think that this api would be the best approach with international names, but likely not the case for usernames or other names not in its data
  - LLM approach
    - I really wanted this to work. The appeal was all there with task flexibility restricted to the prompting, and name flexibility laced within the ideally billion plus parameters in the model. Unfortunately, given the resources that I had, the runs on this function didn't follow the prompt, as they were likely outweighed by the the number of tokens that the email occupied. This could be seen in that many of the outputs were simply trying to continue the email. Another indication was that the responses that did include gender were almost all male, showing some understanding of the prompt, but still not accuracy. I think that in future this will be the way to go, but in order for that to happen I will need to be able to use a larger model, train for longer, and then once those results are more reflective of sender and recipient gender, begin finetuning and merging send and receive functions.

### Methodology 'Wishlist' for Further Research
- Bigger models: This will help to better improve the llm-based methodology
- Gender Identified Enron Corpus: The Prabhakaran et al study said this was freely available online, but I did a lot of searching with no luck. Having this dataset would greatly validate the riveter analysis and greatly increase the number of rows, and thus personas that could be analyzed.
- Unlimited access to a genderizer with confidence prediction: I only used the namesor apion the riveter stuff since my api key didn’t have enough space to use on the larger dataset. As I said in the previous section I think that it performs pretty well on names. When including other naming conventions I would append confidence score to my llm genderizer wishlist.
- More colab gpu time: More time with the colab gpu would significantly increse the potential breadth and performance of this project, allowing for more rows to be analyzed and bigger models to be used in the llm methodology.

### What I Could Have Done Differently
- I could have done a hybrid approach where I use nltk or regex to get a filtered down version of the messages and then and used the model as a double check method to genderize or message trim.
- For gender prediction I could try to finetune a model like we did with the BERT genre labellings in a previous pset. The context window would likely be too small for the emails though, so I would likely have to do a batching approach where I save the best guesses for subject and recipient across model calls.
- I should have used cuml for scikit learn portion of the tokenization approach. PCA on 500k data points would not have gone very smoothly on my cpu, but my methodology didn't account for actually looking at the whole dataset.