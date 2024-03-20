import re
import numpy as np
import pandas as pd
import streamlit as st
from gliner import GLiNER
from collections import Counter
st.set_page_config(layout="wide")
st.title("CMU Fall'24 Housing Matcher")

@st.cache_resource()
def load_model():
  return GLiNER.from_pretrained("urchade/gliner_base")

@st.cache_data
def get_keywords(_model, data, labels):
  results = []
  for i in data:
    entities = _model.predict_entities(i, labels)
    results += [x['text'] for x  in entities]
  return results

gsheet_url = 'https://docs.google.com/spreadsheets/d/1AGXlQ83V8cZeLW4TDcWOrdjofQxC43rBT7OcisklsMs/export?format=csv&gid=316309778'
df_read = pd.read_csv(gsheet_url)
df_read.drop('Timestamp', inplace=True, axis=1)
df_read = df_read.apply(lambda x: x.str.strip())

df = df_read[df_read.iloc[:,4].apply(lambda x: not str(x).startswith('No'))]
status = df.iloc[:,4].fillna(value='').str.split(pat='! ',n=1).apply(lambda x: x[-1])
df['Roommate Status'] = status

remove_cols = ['Planning to attend', 'Any medical conditions/special needs/allergies',\
              'Hobbies and Interests Outside School (Sports, Music, TV Shows/Movies)',\
              'Estimated Arrival at Pittsburgh']
cols_list = [j for j in df.columns[6:-4].tolist() if j not in remove_cols] + [df.columns[-1]]

conditions = '('
for i in cols_list:
  df[i] = df[i].str.title()
  choices = st.multiselect(i, df[i].dropna().unique())
  tmp_condition = ''
  for choice in choices:
    tmp_condition += f'(df["{i}"] == "{choice}")|'
  if tmp_condition:
    tmp_condition = f'({tmp_condition[:-1]})'
    conditions += tmp_condition + '&'

ner_labels = [["hobby", "activity", "sport"], ['quality', 'behaviour', 'skills'], ['quality', 'behaviour', 'skills']]
for idx, nlp_col in enumerate([df.columns[12]] + df.columns[-3:-1].tolist()):
  model = load_model()
  choices_list = get_keywords(model,df[nlp_col].dropna(), ner_labels[idx])
  choices_list = [word.lower().title() for word, word_count in Counter(choices_list).most_common(25)]
  choices = st.multiselect(nlp_col, choices_list)
  tmp_condition = ''
  for choice in choices:
    tmp_condition += f'(df["{nlp_col}"].str.lower().str.contains("{choice.lower()}", regex=False, na=False))|'
  if tmp_condition:
    tmp_condition = f'({tmp_condition[:-1]})'
    conditions += tmp_condition + '&'
conditions = conditions[:-1] + ')'

if st.button('Find roommates!'):
  if conditions != ')':
    #st.write(conditions)
    result_df = df[eval(conditions)]
    result_df = result_df[['Name', 'Gender', 'Program Name', 'CMU Campus', 'Roommate Status', \
               'Home City', 'Home Country', 'Mobile Number (with country code)', \
               'Email address', 'LinkedIn', 'Hobbies and Interests Outside School (Sports, Music, TV Shows/Movies)',\
               'Perks of having you as a roommate 😄', 'Any specific requirements or expectations?']]
    st.dataframe(result_df, hide_index=True)
  else:
    st.write('**Toggle atleast 1 filter to match!**')
