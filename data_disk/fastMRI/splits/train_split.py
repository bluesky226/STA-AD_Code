import os 
import random

csv_file = input("Enter the path to the csv file: ")
with open(csv_file, 'r') as f:
    train_files = f.readlines()


train_files = [x.strip() for x in train_files]



ratio = 0.83 

random.shuffle(train_files)

train_files = train_files[:int(ratio*len(train_files))]
val_files = train_files[int(ratio*len(train_files)):]

print('Using a ratio of ', ratio, ' for train and validation files')
print('The number of training files is: ', len(train_files))
print('The number of validation files is: ', len(val_files))


with open('./train_s1_split.csv', 'w') as f:
    for item in train_files:
        
        f.write("%s\n" % item)
with open('./val_s1_split.csv', 'w') as f:
    for item in val_files:
        
        f.write("%s\n" % item)


print('The path to the train files is: ', os.path.abspath('./train_s1_split.csv'))
print('The path to the validation files is: ', os.path.abspath('./val_s1_split.csv'))
