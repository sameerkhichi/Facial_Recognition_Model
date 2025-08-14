import os

#moving the dataset images to the negative data folder
NEG_PATH = os.path.join('data', 'negative')

try:
    for directory in os.listdir('lfw'):
        for file in os.listdir(os.path.join('lfw', directory)):
            EX_PATH = os.path.join('lfw', directory, file)
            NEW_PATH = os.path.join(NEG_PATH, file)
            os.replace(EX_PATH, NEW_PATH)
except FileNotFoundError as e:
    print("No data to move")




