# -*- coding: utf-8 -*-
# 모듈 import 전 개발 환경에서 Mecab(konlpy) 및 SentenceTransformer 설치 필요
# Colab 환경에서 Mecab 설치 :  https://sosomemo.tistory.com/72
# SentenceTransformer : $pip install sentence_transformers

import torch 
from sentence_transformers import SentenceTransformer, util 

from konlpy.tag import Mecab
from collections import deque
from itertools import combinations
import requests
from bs4 import BeautifulSoup
import re

### 1) embedding simiarity 구현
def metric_embed(content, title, output, embed_model):
  # try:
  content_embedding = embed_model.encode(content)
  title_embedding = embed_model.encode(title)
  output_embedding = embed_model.encode(output)
  # except RuntimeError:

  sim_title = util.pytorch_cos_sim(title_embedding, output_embedding)[0]
  sim_content = util.pytorch_cos_sim(content_embedding, output_embedding)[0]
  return (sim_title.item()+ sim_content.item())/2 # 평균 출력


### 2) Rouge score 구현한 코드
### 특수문자 전처리-> 불용어 크롤링 -> Rouge class로 unigram, bigram, skipbigram 구현 

# 특수문자 전처리
def clean_text(input):
  text_rmv = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', input)
  # print(text_rmv)
  return text_rmv

#불용어 크롤링  
def crawl_stopwords():
    url = "https://www.ranks.nl/stopwords/korean"
    response = requests.get(url, verify = False)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text,'html.parser')
        content = soup.select_one('#article178ebefbfb1b165454ec9f168f545239 > div.panel-body > table > tbody > tr')
        stop_words=[]
        for x in content.strings:
            x=x.strip()
            if x:
                stop_words.append(x)
        print(f"# Korean stop words: {len(stop_words)}")
    else:
        print(response.status_code)

    return set(stop_words)
stopwords = crawl_stopwords()
# Rouge class : unigram, bigram, wholegram 등을 불용어 제거하여 attribute로 할당
class Rouge:
  def __init__(self, text):
    self.text = text
    self.unigrams = self.to_gram(self.text,1)
    self.bigrams = self.to_gram(self.text, 2)
    self.skipbigrams= self.skipbi(self.text)
    self.wholegrams = self.unigrams + self.skipbigrams
  

  def to_gram(self,sen, n):
    lis = []  
    queue = deque(maxlen=n)
    for w in sen:
      if w not in stopwords: ## 크롤링 사용 stopword 제거하기
        queue.append(w)
        if len(queue) == n:
          # if n==1:
            lis.append(tuple(queue))
    return lis
  def skipbi(self, sen):
    iter = combinations(sen,2)
    return list(iter)


# text 입력을 받을 시 Rouge instance로 return

def text_to_gram_instance(text):
  mecab = Mecab()
  text=clean_text(text)
  morphs1=mecab.morphs(text) # 형태소로 분리
  return Rouge(morphs1) #

# 두 개의 gram list를 입력으로 받아 recall, precision, F1 score 계산하는 함수
def scores(reference, output): 
  # output 기준 겹치는 gram 수 
  try:
    intersect=set(reference)&set(output) 
    recall = len(intersect)/len(output) # 
    precision = len(intersect)/len(reference) # title(refrenece) 기준
    f1_score = recall*precision*2/(recall + precision)
    return recall, precision, f1_score
  except ZeroDivisionError:
    return 0, 0, 0

def metric_rouge(content, title, output):
  
  t, o =  text_to_gram_instance(title) , text_to_gram_instance(output)

  # print('------ROUGE-1 SCORE-------')
  title_grams, output_grams = t.unigrams,  o.unigrams


  # 1. F1_title
  rec, pre, f1 = scores(title_grams, output_grams)
  # print('** TITLE vs OUTPUT scores')
  # print(f'recall is {rec:.4f}')
  # print(f'precision is {pre:.4f}')
  # print(f'f1 score is {f1:.4f}')
  return f1

def metric(content, title, output, embed_model):
  sim_score = metric_embed(content, title, output, embed_model)
  rouge_score = metric_rouge(content, title, output)
  return (sim_score+rouge_score)/2
