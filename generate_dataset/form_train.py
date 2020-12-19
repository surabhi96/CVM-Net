import numpy as np 
import os

file = open(os.getcwd()+os.sep+"index.txt","r") 
Counter = 0
  
# Reading from file 
Content = file.read() 
CoList = Content.split("\n") 
  
for i in CoList: 
    if i: 
        Counter += 1
print(Counter)
ind = np.arange(Counter)

train_test = ind
with open('train-test.csv', 'w') as f:
    for id in train_test:
        f.write('satellite/'+str(id)+'.jpg,streetview/'+str(id)+'.jpg,streetview/'+str(id)+'.jpg')
        f.write('\n')

np.random.shuffle(ind)

div = int(0.66*Counter)

train = ind[:div]
test = ind[div:Counter]

with open('train-19zl.csv', 'w') as f:
    for id in train:
        f.write('satellite/'+str(id)+'.jpg,streetview/'+str(id)+'.jpg,streetview/'+str(id)+'.jpg')
        f.write('\n')
with open('test-19zl.csv', 'w') as f:
    for id in test:
        f.write('satellite/'+str(id)+'.jpg,streetview/'+str(id)+'.jpg,streetview/'+str(id)+'.jpg')
        f.write('\n')

