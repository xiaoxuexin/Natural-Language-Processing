import re
import csv
with open('email-outputs.csv', 'w') as csvfilew:
    filewriter = csv.writer(csvfilew, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['html', 'email'])
def regex_search(sentence):
    result = re.search(r'([a-zA-Z0-9.!#$%&\'*+-/=?^_`{|}~\"]+)\s*(\[\s*at\s*\ *]|@|\|\s*at\s*\||\/\s*at\s*\/|\\\s*at\s*\\|\ *at\ )+\s*([a-zA-Z0-9\-]+)\s*(\[\s*dot\s*\]|\.|\|\s*dot\s*\||\/\s*dot\s*\/|\\\s*dot\s*\\|\ \s*dot\s*\ )\s*([a-zA-Z0-9\-]+)', sentence)
    if result:
        with open('email-outputs.csv', 'a') as csvfilew:
            filewriter = csv.writer(csvfilew, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow([sentence,result.group(1) + "@" + result.group(3) + "." + result.group(5)])
    else:
        with open('email-outputs.csv', 'a') as csvfilew:
            filewriter = csv.writer(csvfilew, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow([sentence, 'None'])
with open('webpages.csv', 'r') as csvfiler:
    filereader = csv.reader(csvfiler, delimiter = ',')
    first_row = next(filereader)
    for row in filereader:
        regex_search(row[0])

