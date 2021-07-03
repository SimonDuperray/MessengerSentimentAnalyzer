from math import *
from tkinter import *
from PIL import ImageTk,Image 
import matplotlib.pyplot as plt

def not_all_the_time():
   import nltk, re, pprint, json, datetime, ftfy, pandas, numpy as np
   nltk.download([
                  "words",
                  "punkt",
                  "vader_lexicon",
                  "stopwords"
   ])
   from nltk.sentiment import SentimentIntensityAnalyzer
   from googletrans import Translator
   from functools import reduce
   sia = SentimentIntensityAnalyzer()
   translator = Translator()

   def timestamp2date(ts):
      return datetime.datetime.fromtimestamp(ts/1000.0)

   def stopped_words(words_list):
      punctuation = re.compile(r'[-.?!,:;’"”“()|0-9]')
      post_punctuation = []
      tokens = nltk.word_tokenize(words_list)
      for words in tokens:
         word = punctuation.sub("", words)
         if len(word)>0:
            post_punctuation.append(word)

   def print_indent(to_print):
      print(json.dumps(to_print, indent=4))

   def translate_fr2en(txt_fr):
      return translator.translate(txt_fr, dest="en").text

   def get_nb_msgs(list_of_files):
      len_=0
      for fi in list_of_files:
         with open(fi, "r", encoding="utf-8") as outfile:
            len_+=len(json.load(outfile)["messages"])
      return len_

   def get_percent(len_, current):
      return round(100*current/len_, 2)

   def average(lst):
      return reduce(lambda a, b: a+b, lst)/len(lst)

   def create_data_structure(msgs, sorted_msgs, unique_dates):
      for i in range(len(unique_dates)):
         bf_obj = {
            "date": unique_dates[i],
            "msgs": [],
            "polarity": []
         }
         for j in range(len(msgs)):
            if msgs[j]["date"]==unique_dates[i]:
               bf_obj["msgs"].append(msgs[j]["content"])
         sorted_msgs.append(bf_obj)

   def get_polarity(sorted_msgs):
      for month in range(len(sorted_msgs)):
         print(f'Date: {sorted_msgs[month]["date"]}')
         sorted_msgs[month]["polarity"] = []
         for msg in range(len(sorted_msgs[month]["msgs"])):
            sorted_msgs[month]["polarity"].append(sia.polarity_scores(sorted_msgs[month]["msgs"][msg]))

   def remake_polarity(pol_scores, sorted_msgs):
      for k in range(len(sorted_msgs)):
         print(sorted_msgs[k]["polarity"])
         pol_scores.append(sorted_msgs[k]["polarity"])

   def write_data(filename, pol_scores):
      filename+=".json"
      with open(filename, "w") as outfile:
         json.dump(pol_scores, outfile)

   def polarity_decomposer(sorted_msgs):
      for month in range(len(sorted_msgs)):
         neg_list, neu_list, pos_list, compound_list = [], [], [], []
         if "neg" in sorted_msgs[month]["polarity"] and "neu" in sorted_msgs[month]["polarity"] and "pos" in sorted_msgs[month]["polarity"] and "compound" in sorted_msgs[month]["polarity"]:
            sorted_msgs[month]["polarity"]["neg"] = []
            sorted_msgs[month]["polarity"]["neu"] = []
            sorted_msgs[month]["polarity"]["pos"] = []
            sorted_msgs[month]["polarity"]["compound"] = []
         for scores in range(len(sorted_msgs[month]["polarity"])):
            neg_list.append(sorted_msgs[month]["polarity"][scores]["neg"])
            neu_list.append(sorted_msgs[month]["polarity"][scores]["neu"])
            pos_list.append(sorted_msgs[month]["polarity"][scores]["pos"])
            compound_list.append(sorted_msgs[month]["polarity"][scores]["compound"])
         sorted_msgs[month]["pol_means"] = {
            "neg": average(neg_list),
            "neu": average(neu_list),
            "pos": average(pos_list),
            "compound": average(compound_list)
         }

   def list_of_means(sorted_msgs):
      neg_mean, neu_mean, pos_mean, compound_mean = [], [], [], []
      for month in range(len(sorted_msgs)):
         neg_mean.append(sorted_msgs[month]["pol_means"]["neg"])
         neu_mean.append(sorted_msgs[month]["pol_means"]["neu"])
         pos_mean.append(sorted_msgs[month]["pol_means"]["pos"])
         compound_mean.append(sorted_msgs[month]["pol_means"]["compound"])
      return neg_mean, neu_mean, pos_mean, compound_mean


   msgs_s, msgs_a = [], []
   fis = ["message_1.json", "message_2.json", "message_3.json", "message_4.json", "message_5.json","message_6.json"]
   len_ = get_nb_msgs(fis)

   for fi in fis:
      with open(fi, "r", encoding="utf-8") as outfile:
         stocked = json.load(outfile)["messages"]
         len_ = len(stocked)
         for i in range(len_):
            if "content" in stocked[i]:
               cleaned_content = str(ftfy.ftfy(stocked[i]["content"]))
               if stocked[i]["sender_name"]=="SD":
                  if cleaned_content != "Vous avez appelé a." and cleaned_content != "a a manqué votre appel." and "https" not in cleaned_content:
                     msgs_s.append({
                        "date": str(timestamp2date(stocked[i]["timestamp_ms"]).date())[:7],
                        "content": str(ftfy.ftfy(stocked[i]["content"]))
                     })
               if stocked[i]["sender_name"]=="AD":
                  if cleaned_content != "Vous avez appelé s." and cleaned_content != "s a manqué votre appel." and "https" not in cleaned_content:
                     msgs_a.append({
                        "date": str(timestamp2date(stocked[i]["timestamp_ms"]).date())[:7],
                        "content": str(ftfy.ftfy(stocked[i]["content"]))
                     })
            print("data storing: ["+str(get_percent(len_, i+1))+"%] -- ["+str(fi)+"]")


   # get unique dates list
   global unique_dates
   dates, unique_dates = [], []
   for itm in range(len(msgs_s)):
      dates.append(msgs_s[itm]["date"])
   unique_dates = list(dict.fromkeys(dates))[::-1]
   with open("unique_dates.json", "w") as outfile:
      json.dump(unique_dates, outfile)

   sorted_msgs_s, sorted_msgs_a = [], []
   create_data_structure(msgs_s, sorted_msgs_s, unique_dates)
   create_data_structure(msgs_a, sorted_msgs_a, unique_dates)

   # get polarity scores
   get_polarity(sorted_msgs_s)
   get_polarity(sorted_msgs_a)

   # remake polarity scores
   pol_scores_s, pol_scores_a = [], []
   remake_polarity(pol_scores_s, sorted_msgs_s)
   remake_polarity(pol_scores_a, sorted_msgs_a)

   # write data
   write_data("polarity_scores_s", pol_scores_s)
   write_data("polarity_scores_a", pol_scores_a)

   # decompose polarity scores
   polarity_decomposer(sorted_msgs_s)
   polarity_decomposer(sorted_msgs_a)

   # print indent
   for month in range(len(sorted_msgs_s)):
      print_indent(sorted_msgs_s[month]["pol_means"])
   for month in range(len(sorted_msgs_a)):
      print_indent(sorted_msgs_a[month]["pol_means"])

   # create lists of means for all months
   global neg_meanS, neu_meanS, pos_meanS, compound_meanS, neg_meanA, neu_meanA, pos_meanA, compound_meanA
   neg_meanS, neu_meanS, pos_meanS, compound_meanS = list_of_means(sorted_msgs_s)
   neg_meanA, neu_meanA, pos_meanA, compound_meanA = list_of_means(sorted_msgs_a)

   # graphs

   # neg
   fig = plt.figure(figsize=(20, 5))
   plt.title("Négativité - basic")
   plt.plot(unique_dates, neg_meanS, label="s", marker="o")
   plt.plot(unique_dates, neg_meanA, label="a", marker="v")
   plt.xticks(rotation=90)
   plt.legend(loc="upper left")
   plt.show()
   fig.savefig("neg.png", dpi=200)

   # pos
   fig = plt.figure(figsize=(20, 5))
   plt.title("Positivité - basic")
   plt.plot(unique_dates, pos_meanS, label="s", marker="o")
   plt.plot(unique_dates, pos_meanA, label="a", marker="v")
   plt.xticks(rotation=90)
   plt.legend(loc="upper left")
   plt.show()
   fig.savefig("pos.png", dpi=200)

   # neu
   fig = plt.figure(figsize=(20, 5))
   plt.title("Neutral - basic")
   plt.plot(unique_dates, neu_meanS, label="s", marker="o")
   plt.plot(unique_dates, neu_meanA, label="a", marker="v")
   plt.xticks(rotation=90)
   plt.legend(loc="upper left")
   plt.show()
   fig.savefig("neu.png", dpi=200)

   # compound
   fig = plt.figure(figsize=(20, 5))
   plt.title("Compound - basic")
   plt.plot(unique_dates, compound_meanS, label="s")
   plt.plot(unique_dates, compound_meanA, label="a")
   plt.xticks(rotation=90)
   plt.legend(loc="upper left")
   plt.show()
   fig.savefig("compound.png", dpi=200)

   # all
   fig = plt.figure(figsize=(20, 5))
   plt.title("Neg - Pos [Both-basic]")
   plt.plot(unique_dates, pos_meanS, label="s [pos]", marker="o")
   plt.plot(unique_dates, pos_meanA, label="a [pos]", marker="o")
   plt.plot(unique_dates, neg_meanS, label="s [neg]", marker="v")
   plt.plot(unique_dates, neg_meanA, label="a [neg]", marker="v")
   plt.xticks(rotation=90)
   plt.legend(loc="upper left")
   plt.show()
   fig.savefig("comp.png", dpi=200)

not_all_the_time()

def run_pos():
   fig = plt.figure(figsize=(20, 5))
   plt.title("Positivité - basic")
   plt.plot(unique_dates, pos_meanS, label="s", marker="o")
   plt.plot(unique_dates, pos_meanA, label="a", marker="v")
   plt.xticks(rotation=90)
   plt.legend(loc="upper left")
   plt.show()

def run_neu():
   fig = plt.figure(figsize=(20, 5))
   plt.title("Neutral - basic")
   plt.plot(unique_dates, neu_meanS, label="s", marker="o")
   plt.plot(unique_dates, neu_meanA, label="a", marker="v")
   plt.xticks(rotation=90)
   plt.legend(loc="upper left")
   plt.show()

def run_neg():
   fig = plt.figure(figsize=(20, 5))
   plt.title("Négativité - basic")
   plt.plot(unique_dates, neg_meanS, label="s", marker="o")
   plt.plot(unique_dates, neg_meanA, label="a", marker="v")
   plt.xticks(rotation=90)
   plt.legend(loc="upper left")
   plt.show()

def run_compound():
   fig = plt.figure(figsize=(20, 5))
   plt.title("Compound - basic")
   plt.plot(unique_dates, compound_meanS, label="s")
   plt.plot(unique_dates, compound_meanA, label="a")
   plt.xticks(rotation=90)
   plt.legend(loc="upper left")
   plt.show()

gui = Tk()
gui.geometry("200x200")

pos_button = Button(gui, text="pos", command=run_pos)
pos_button.pack()

neu_button = Button(gui, text="neu", command=run_neu)
neu_button.pack()

neg_button = Button(gui, text="neg", command=run_neg)
neg_button.pack()

compound_button = Button(gui, text="compound", command=run_compound)
compound_button.pack()

gui.mainloop()