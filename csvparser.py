"""
Skrypt parsujący dane z formatu csv do pliku json
"""

import csv
import json

userDict = {}

class MovieInfo:
    def __init__(self, movieName, movieScore):
        self.movieName = movieName
        self.movieScore = movieScore

# Reading the CSV and populating the dictionary
with open("Oceny.csv", encoding='utf-8') as fp:
    reader = csv.reader(fp, delimiter=";", quotechar='"')
    data_read = [row for row in reader]

for idxRow, row in enumerate(data_read):
    if idxRow == 0:  # Skip the header row
        continue
    userName = ""
    movieName = ""
    movieScore = ""
    for idxInnerRow, innerRow in enumerate(row):
        if idxInnerRow == 0:
            continue
        if idxInnerRow == 1:
            userName = innerRow
            userDict[innerRow] = []
            continue
        if idxInnerRow % 2 == 0:  # Movie name on even indices
            movieName = innerRow
        else:  # Movie score on odd indices
            movieScore = innerRow
            if movieScore and movieName:  # Ensure that neither movie name nor score is empty
                userDict[userName].append(MovieInfo(movieName=movieName, movieScore=float(movieScore)))

# Using json library to write the dictionary to a JSON file
with open('dataset.json', 'w') as file:
    json_data = {key: {mi.movieName: mi.movieScore for mi in values} for key, values in userDict.items()}
    json.dump(json_data, file, indent=4)
