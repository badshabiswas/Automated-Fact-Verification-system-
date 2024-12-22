import csv
import numpy as np
import pandas as pd

# Specifying the input and output file paths
input_file_path = '/scratch/mbiswas2/CS 580/train.txt'
output_file_path = '/scratch/mbiswas2/CS 580/train.csv'  # Change the file extension to CSV

# Initializing lists to store the modified content
updatedReview = []


# strings to remove and a corresponding list for removed strings
label = ['+1', '-1']
labelRemoved = []



# Open the input file for reading
with open(input_file_path, 'r') as input_file:
    # Read the content of the input file
    content = input_file.read()

    # Splitting the content into lines
    lines = content.split('\n')

    # Looping through each line and remove the specified strings
    for line in lines:
        updatedLine = line  # Initializing the modified line
        for string_to_remove in label:
            updatedLine = updatedLine.replace(string_to_remove, '')

        # Appending the modified line and the removed string to their respective lists
        updatedReview.append(updatedLine)
        labelRemoved.append(', '.join([string_to_remove for string_to_remove in label if string_to_remove in line]))

# Writing the data to a CSV file
with open(output_file_path, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)

    # Write the header row
    csv_writer.writerow(['Reviews', 'Ratings'])

    # Write the data row by row
    for updatedLine, updatedString in zip(updatedReview, labelRemoved):
        csv_writer.writerow([updatedLine, updatedString])
