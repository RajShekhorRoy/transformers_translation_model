from torchtext.data.functional import generate_sp_model
import csv
import pandas as pd
# csv_file = csv.reader("/Users/rajshekhorroy/Documents/bangla_sent.csv.numbers")
unique_word = []
# with open("/Users/rajshekhorroy/Documents/bangla_sent.csv.numbers", newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     for row in spamreader:
#         for a in row.split():
#          unique_word.append(a)
# csv_file = "/Users/rajshekhorroy/Documents/bangla_sent.csv.numbers"  # Replace with your file path
# df = pd.read_csv(csv_file)
#
# print(df)
generate_sp_model('./bangla_cleaned.csv', vocab_size=23456,model_prefix='spm_user')

# sp_model = load_sp_model("./spm_user.model")
# sp_id_generator = sentencepiece_numericalizer(sp_model)
# tokenizer = sp_id_generator(['একটি গোলাপী জামা পরা বাচ্চা মেয়ে একটি বাড়ির প্রবেশ পথের সিঁড়ি বেয়ে উঠছে'
# ,'একটি মেয়ে শিশু একটি কাঠের বাড়িতে ঢুকছে'
# ,'একটি বাচ্চা তার কাঠের খেলাঘরে উঠছে ।'
# ,'ছোট মেয়েটি তার খেলার ঘরের সিড়ি বেয়ে উঠছে'])
data=[]
# print(list(tokenizer))

# # Open the file and read line by line and remove special characters
# dat = pd.read_csv("/Users/rajshekhorroy/Documents/english_to_bengali_2.csv",header=None)
# for line_number, line in dat.iterrows():
#
#     if line_number == 0:
#         continue
#     parts = str(line.values[0]).replace("?","").replace("।","").replace(",","").replace("'","").replace('"','')
#
#     # Take only the first column (assuming the desired data is in the first column)
#     data.append(parts)
#
# df = pd.DataFrame(data)
# df.to_csv('bangla_cleaned.csv', index=False,header=False)
#16017 unique words
# unique_word_index=[]
# cleaned_data=pd.read_csv("./bangla_cleaned.csv",header=None)
# for line_number, line in cleaned_data.iterrows():
#     for values in str(line.values[0]).split():
#         unique_word_index.append(values)
# print("done")

