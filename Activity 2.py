#exercises 10.03
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def sigma(numbers):
    sums = []
    total = 0
    for i in numbers:
        total += i
        sums.append(total)
    print sums

#sigma(numbers)

#exercises 10.05

nom = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def chop(nom):
    del nom[0]
    del nom[-1]
    print nom

#print chop(nom)


def middle(nom):
    t = nom[1:-1]
    return t

#print middle(nom)

#Exercises 10.06
listOne = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#listOne = [7, 3, 6, 10, 2, 8, 4, 1, 9, 5]


def is_sorted(listOne):
    return sorted(listOne) == listOne

#print is_sorted(listOne)

#Exercise 10.07

stringOne = "events"
stringTwo = "steven"


def is_anagram(stringOne, stringTwo):
    print sorted(stringOne) == sorted(stringTwo)

#is_anagram(stringOne, stringTwo)

#Exercise 10.08

import random

NUMBER_OF_STUDENTS = 23
TRIALS = 1000


def has_duplicates(my_list):
    i = 0
    while i < len(my_list):
        if my_list.count(my_list[i]) > 1:
            return True
        elif i == (len(my_list) - 1):
            return False
        i += 1


def generate_random_birthdays():
    return [random.randint(1, 365) for student in range(NUMBER_OF_STUDENTS)]


def stats(TRIALS):

    duplicate_count = 0
    for i in range(TRIALS):
        if has_duplicates(generate_random_birthdays()):
            duplicate_count += 1
    print "In %d classrooms with %d students, %.1f%% had students\
 with duplicate birthdays." % (TRIALS, NUMBER_OF_STUDENTS, (float(duplicate_count) / TRIALS) * 100)


#stats(TRIALS)

#Exercise 10.09

items = [1, 1, 2, 4, 5, 6, 7, 8, 8, 9, 10, 7]


def remove_duplicates(items):
    return list(set(items))

#print remove_duplicates(items)

#Exercise 10.10

with open('words.txt') as fd:
    words = fd.read().split()


def method_one(words):
    wordList = []
    for line in words:
        line = line.strip()
        wordList.append(line)
    print len(wordList)
    print wordList[:10]

#method_one(words)

def method_two(words):
    wordList = []
    for line in words:
        line = line.strip()
        wordList += [line]
    print len(wordList)
    print wordList[:10]

#method_two(words)

#Exercises 10.11

with open('words.txt') as fd:
    word_list = fd.read().splitlines()


def bisect(myWord, myList):
    original = myList
    while True:
        middle = len(myList) / 2
        if myWord > myList[middle]:
            myList = myList[middle:]
        elif myWord < myList[middle]:
            myList = myList[:middle]
        elif myWord == myList[middle]:
            return original.index(myWord)

        if len(myList) == 1:
            if myWord != myList[:]:
                return None
            else:
                return original.index(myWord)


#print bisect("danger", word_list)

#Exercise 10.12

with open('words.txt') as fd:
    word_list = fd.read().splitlines()

word_dict = {word: None for word in word_list}

def find_rev_pairs(word_dict):
    for word in word_dict:
        if word[::-1] in word_dict:
            print word, word[::-1]

#find_rev_pairs(word_dict)

#Exercise 10.13

with open('words.txt') as fd:
    word_list = fd.read().splitlines()

word_dict = {word: None for word in word_list}


def split_word(word):
    word1 = word[::2]
    word2 = word[1::2]
    return (word1, word2)


def find_interlocked():
    for word in word_dict:
        split0 = split_word(word)[0]
        split1 = split_word(word)[1]
        if (split0 in word_dict and split1 in word_dict):
                print word, split0, split1

#find_interlocked()


def split_word2(word, i):
    split0 = word[i::3]
    split1 = word[i + 1::3]
    split2 = word[i + 2::3]
    return (split0, split1, split2)


def find_3way():
    answer = []
    for word in word_dict:
        for i in range(0, 3):
            split_ = split_word2(word, i)
            if (split_[0] in word_dict and
                split_[1] in word_dict and
                split_[2] in word_dict):
                    answer.append((word,
                           split_[0],
                           split_[1],
                           split_[2]))
    return answer

#print find_3way()
#print "Done"

#Exercise 11.01

import uuid

with open('words.txt') as fd:
    words = fd.read().splitlines()

result = dict()


def dictionary():
    for line in words:
        result[line] = uuid.uuid4()
    return result

#print dictionary()

#Exercise 11.02

def histogram(word):
    dictionary = dict()
    for character in word:
        dictionary[character] = 1 + dictionary.get(character, 0)
    return dictionary

#print histogram('antidisestablishmentarianism')

#Exercise 11.03

def histogram(word):
    dictionary = dict()
    for letter in word:
        dictionary[letter] = 1 + dictionary.get(letter, 0)
    return dictionary


def print_hist(histogram):
    histoList = histogram.keys()
    histoList.sort()
    for letter in histoList:
        print letter, histogram[letter]

h = histogram('parrot')
#print_hist(h)

#Exercise 11.04

def reverse_lookup(dictionary, value):
    results = []
    for key in dictionary:
        if dictionary[key] == value:
            results.append(key)
    return results


def histogram(word):
    dictionary = dict()
    for letter in word:
        dictionary[letter] = 1 + dictionary.get(letter, 0)
    return dictionary

h = histogram('parrot')
k = reverse_lookup(h, 2)
#print k

#Exercise 11.05

def histogram(word):
    dictionary = dict()
    for letter in word:
        dictionary[letter] = 1 + dictionary.get(letter, 0)
    return dictionary


def invert_dict(d):
    inv = dict()
    for key in d:
        val = d[key]
        inv.setdefault(val, [])
        inv[val].append(key)
    return inv

hist = histogram('parrot')
#print hist
inv = invert_dict(hist)
#print inv

#Exercise 11.06

known = {0: 0, 1: 1}


def fibonacci(n):
    if n in known:
        return known[n]
    else:
        res = fibonacci(n - 1) + fibonacci(n - 2)
    known[n] = res
    return res

#def fibonacci(n):
#    if n == 0:
#        return 0
#    elif n == 1:
#        return 1
#    else:
#        return fibonacci(n - 1) + fibonacci(n - 2)

#print fibonacci(40)

#Exercise 11.07

cache = {}

def ackermann(m, n):
    """Computes the Ackermann function A(m, n)

    See http://en.wikipedia.org/wiki/Ackermann_function

    n, m: non-negative integers
    """
    if m == 0:
        return n+1
    if n == 0:
        return ackermann(m-1, 1)
    try:
        return cache[m, n]
    except KeyError:
        cache[m, n] = ackermann(m-1, ackermann(m, n-1))
        return cache[m, n]

#print ackermann(3, 4)
#print ackermann(3, 6)
#copyright http://www.greenteapress.com/thinkpython/code/ackermann_memo.py

#Exercise 11.08












#Exercise 11.09

listOne = [3, 5, 6, 7, 5, 4, 6,]


def has_dups(myList):
    dictionary = {}
    for item in myList:
        dictionary[item] = 1 + dictionary.get(item, 0)
        if dictionary[item] > 1:
            return True
    return False

print has_dups(listOne)

#Exercise 11.10

import rotate

with open('words.txt') as fd:
    word_list = fd.read().splitlines()

word_dict = {word: None for word in word_list}


def find_rot_pairs():
    final_list = []
    for word in word_dict:
        for i in range(1, 26):
            if rotate.rotate_word(word, i) in word_dict:
                final_list.append((word, i, rotate.rotate_word(word, i)))
    final_list.sort()
    for pair in final_list:
        print pair

#find_rot_pairs()

#Exercise 11.11











#Exercise 12.01

def sum_all(*args):
    return sum(args)


#print sum_all(1, 2, 3)
#print sum_all(1, 2, 3, 4, 5)
#print sum_all(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


#Exercise 12.02

import random

with open('words.txt') as fd:
    words = fd.read().splitlines()


def sort_by_length(words):
    t = []
    for word in words:
        t.append((len(word), word))
        t.sort(reverse=True)
    res = []
    for length, word in t:
        res.append(word)
    return res

#print sort_by_length(words)


#Exercise 12.03

text = 'Hello There im currently sitting'


def make_dict(x):
    dictionary = {}
    for letter in x:
        dictionary[letter] = 1 + dictionary.get(letter, 0)
    return dictionary


def most_frequent(text):
    letters = [letter.lower() for letter in text if letter.isalpha()]
    dictionary = make_dict(letters)
    result = []
    for key in dictionary:
        result.append((dictionary[key], key))
    result.sort(reverse=True)
    for count, letter in result:
        print letter, count

#most_frequent(text)


#Exercise 12.04















#Exercise 12.05

















#Exercise 12.06

def make_word_dict():

    d = dict()
    fin = open('words.txt')
    for line in fin:
        word = line.strip().lower()
        d[word] = word
    for letter in ['a', 'i', '']:
        d[letter] = letter
    return d


memo = {}
memo[''] = ['']


def is_reducible(word, word_dict):

    if word in memo:
        return memo[word]
    res = []
    for child in children(word, word_dict):
        t = is_reducible(child, word_dict)
        if t:
            res.append(child)
    memo[word] = res
    return res


def children(word, word_dict):
    res = []
    for i in range(len(word)):
        child = word[:i] + word[i+1:]
        if child in word_dict:
            res.append(child)
    return res


def all_reducible(word_dict):
    res = []
    for word in word_dict:
        t = is_reducible(word, word_dict)
        if t != []:
            res.append(word)
    return res


def print_trail(word):
    if len(word) == 0:
        return
    print word,
    t = is_reducible(word, word_dict)
    print_trail(t[0])


def print_longest_words(word_dict):
    words = all_reducible(word_dict)

    t = []
    for word in words:
        t.append((len(word), word))
    t.sort(reverse=True)
    for length, word in t[0:5]:
        print_trail(word)
        print '\n'


if __name__ == '__main__':
    word_dict = make_word_dict()
    #print_longest_words(word_dict)


#Exercise 13.01

from string import punctuation, whitespace

book = 'origin.txt'

with open(book, 'r') as fd:
    words = fd.read().split()

#remove punctuation, whitespace, uppercase
def clean(word):
    cleansed = ''
    for char in word:
        if ((char in punctuation) or (char in whitespace)):
            pass
        else:
            cleansed += char.lower()
    return cleansed

#print "{} has {} 'words'".format(book, len([clean(word) for word in words]))


#Exercise 13.02

from string import punctuation, whitespace

origin = 'origin.txt'
huck = 'huck.txt'
frank = 'frank.txt'
great = 'great.txt'
meta = 'meta.txt'
sherlock = 'sherlock.txt'
tale = 'tale.txt'

def words(book):
    list_ = []
    flag = False
    signal = "*** START OF"
    for line in book:
        if flag == True:
            for word in line.split():
                list_.append(word)
        elif (signal in line) and (flag == False):
            flag = True
        else:
            pass
    return list_

def clean(word):
    result = ''
    for letter in word:
        if (letter in whitespace) or (letter in punctuation):
            pass
        else:
            result += letter.lower()
    return result

def histogram(data):
    hist = {}
    for word in data:
        hist[word] = hist.get(word, 0) + 1
    return hist

books = [origin, huck, frank, great, meta, sherlock, tale]

def stats():
    for book in books:
        book = open(book, 'r')
        print "Stats for %s:" % book.name
        data = [clean(word) for word in words(book)]
        book.close()
        print "  Total: %s" % len(data)
        print "  Unique: %s" % len(histogram(data))

#stats()

#Exercise 13.03

from string import punctuation, whitespace

origin = 'origin.txt'
huck = 'huck.txt'
frank = 'frank.txt'
great = 'great.txt'
meta = 'meta.txt'
sherlock = 'sherlock.txt'
tale = 'tale.txt'

books = [origin, huck, frank, great, meta, sherlock, tale]

def words(book):
    list_ = []
    flag = False
    signal = "*** START OF"
    op = open(book, 'r')
    for line in op:
        if flag == True:
            for word in line.split():
                list_.append(word)
        elif (signal in line) and (flag == False):
            flag = True
        else:
            pass
    op.close()
    return list_


def clean(word):
     result = ''
     for char in word:
         if (char in whitespace) or (char in punctuation):
             pass
         else:
             result += char.lower()
     return result

def histogram(data):
     hist = {}
     for word in data:
         if word == '':
             pass
         else:
             hist[word] = hist.get(word, 0) + 1
     return hist

def main():
    for book in books:
        data = [clean(word) for word in words(book)]
        print "Stats for %s:" % book
        hist = histogram(data)
        top20 = []
        for key in hist:
            top20.append([hist[key], key])
        top20.sort(reverse=True)
        for i in range(0, 20):
            print "  %s) %s %s" % (i + 1, top20[i][1], top20[i][0])

        print "\n"

#main()


#Exercise 13.04

from string import punctuation, whitespace

origin = 'origin.txt' # Origin of Species, 1859
huck = 'huck.txt' # Huck Finn, 1884
don = 'don.txt' # Don Quixote, 1605
great = 'great.txt' # Expectations, 1860
meta = 'meta.txt' # morphisis, 1915
sherlock = 'sherlock.txt' # 1887
divine = 'divine.txt' # Comedy, 1308
journey = 'journey.txt'  # to the center of the earth, 1864

word_file = 'words.txt'
books = [origin, huck, don, great, meta, sherlock, divine, journey]

def words(book):
    list_ = []
    flag = False
    signal = "*** START OF"
    for line in book:
        if flag == True:
            for word in line.split():
                list_.append(word)
        elif (signal in line) and (flag == False):
            flag = True
        else:
            pass
    return list_

def clean(word):
    result = ''
    for char in word:
        if (char in whitespace) or (char in punctuation):
            pass
        elif not char.isalpha():
            pass
        else:
            result += char.lower()
    return result

def stats():
    for book in books:
        book_words = set([clean(word) for word in words(open(book, 'r'))])
        words_ = set([word for word in open(word_file, 'r')])
        print "Stats for %s" % open(book, 'r').name
        print "  There are %s non-listed words." % len(book_words - words_)

#stats()

#print "\n\nThe words not in the word list for origin.txt:"
#print set([clean(word) for word in words(open(origin, 'r'))]) - \
#      set([word for word in open(word_file, 'r')])

#Exercise 13.05

import random

t = ['a', 'a', 'b']

def hist(x):
    hist = {}
    for item in x:
        hist[item] = hist.get(item, 0) + 1
    return hist

hist = hist(t)

def choose_from_hist(hist):
    list_ = []
    for key in hist:
        for i in range(0, hist[key]):
            list_.append(key)
    return random.choice(list_)

def stats():
    a = 0
    b = 0
    for i in range(0, 10000):
        if choose_from_hist(hist) == 'a':
            a += 1
        else:
            b += 1
    print "a: %.5f" % (a / 10000.0), "b: %.5f" % (b / 10000.0)

#stats()

#Exercise 13.06













#Exercise 13.07









#Exercise 13.08










#Exercise 13.09








#Exercise 14.01

import os

cwd = os.getcwd()
names = []

def walk(directory):
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            names.append(path)
        else:
            walk(path)
    print names

#walk(cwd)



#Exercise 14.02

import sys


def sed(pattern, replace, source, dest):
    try:
        fin = open(source, 'r')
        fout = open(dest, 'w')

        for line in fin:
            line = line.replace(pattern, replace)
            fout.write(line)

        fin.close()
        fout.close()
    except:
        print 'Something went wrong.'


def main(name):
    pattern = 'pattern'
    replace = 'replacendum'
    source = name
    dest = name + '.replaced'
    sed(pattern, replace, source, dest)


if __name__ == '__main__':
    main(*sys.argv)


#Exercise 14.03











#Exercise 14.04







#Exercise 14.05







#Exercise 14.06


import urllib

zipcode = '02492'

url = 'http://uszip.com/zip/' + zipcode
conn = urllib.urlopen(url)

for line in conn.fp:
    line = line.strip()
    if 'Population' in line:
        print line
    if 'Longitude' in line:
        print line
    if 'Latitude' in line:
        print line

#conn.close()


#Exercise 15.01

import math


class Point(object):
    """Represents a point in 2d space."""

point_one = Point()
point_two = Point()

point_one.x, point_one.y = 6.0, 1.0
point_two.x, point_two.y = 2.0, 6.0


def distance(p1, p2):
    """Returns the distance between two points in 2d space."""
    delta_x = p2.x - p1.x
    delta_y = p2.y - p1.y
    return math.sqrt(delta_x ** 2 + delta_y ** 2)

#print "The distance between point one at (%g,%g)" % (point_one.x, point_one.y),
#print "and point two at (%g,%g)" % (point_two.x, point_two.y),
#print "is %.3f" % distance(point_one, point_two)


#Exercise 15.02

class Point(object):
    """Represents a point in 2d space"""


class Rectangle(object):
    """Represents a rectangle in 2d space"""

rectangle = Rectangle()

bottom_left = Point()
bottom_left.x = 3.0
bottom_left.y = 5.0

top_right = Point()
top_right.x = 5.0
top_right.y = 10.0

rectangle.corner1 = bottom_left
rectangle.corner2 = top_right

dx = 5.0
dy = 12.0


def move_rectangle(rectangle, dx, dy):
    """Takes a rectangle and moves it to the values of dx and dy."""
    print ("The rectangle started with bottom left corner at (%g,%g)"
           % (rectangle.corner1.x, rectangle.corner1.y)),
    print ("and top right corner at (%g,%g)."
           % (rectangle.corner2.x, rectangle.corner2.y)),
    print "dx is %g and dy is %g" % (dx, dy)
    rectangle.corner1.x = rectangle.corner1.x + dx
    rectangle.corner2.x = rectangle.corner2.x + dx
    rectangle.corner1.y = rectangle.corner1.y + dy
    rectangle.corner2.y = rectangle.corner2.y + dy
    print ("It ended with a bottom left corner at (%g,%g)"
           % (rectangle.corner1.x, rectangle.corner1.y)),
    print ("and a top right corner at (%g,%g)"
           % (rectangle.corner2.x, rectangle.corner2.y))

#move_rectangle(rectangle, dx, dy)


#Exercise 15.03

import copy


class Point(object):
    """Represents a point in 2d space"""


class Rectangle(object):
    """Represents a rectangle in 2d space"""

rectangle = Rectangle()

bottom_left = Point()
bottom_left.x = 3.0
bottom_left.y = 5.0

top_right = Point()
top_right.x = 5.0
top_right.y = 10.0

rectangle.corner1 = bottom_left
rectangle.corner2 = top_right

dx = 5.0
dy = 12.0


def move_rectangle(rectangle, dx, dy):
    new_rectangle = copy.deepcopy(rectangle)
    print ("Original: (%g,%g)" % (rectangle.corner1.x, rectangle.corner1.y)),
    print ("(%g,%g)" % (rectangle.corner2.x, rectangle.corner2.y))
    new_rectangle.corner1.x = new_rectangle.corner1.x + dx
    new_rectangle.corner2.x = new_rectangle.corner2.x + dx
    new_rectangle.corner1.y = new_rectangle.corner1.y + dy
    new_rectangle.corner2.y = new_rectangle.corner2.y + dy
    print ("New: (%g,%g)" % (new_rectangle.corner1.x,
           new_rectangle.corner1.y)),
    ("(%g,%g)" % (new_rectangle.corner2.x, new_rectangle.corner2.y))

#move_rectangle(rectangle, dx, dy)

#Exercise 15.04








#Exercise 16.01

class Time(object):
    """ represents the time of day.
    attributes: hour, minute, second"""

time = Time()
time.hour = 11
time.minute = 59
time.second = 30


def print_time(time):
    print "%.2d:%.2d:%.2d" % (time.hour, time.minute, time.second)

#print_time(time)

#Exercise 16.02

import time
import datetime

class Time(object):
    """Time object based on datetime.datetime describes time in 24hr format"""
    def __init__(self, year=2000, month=1, day=1, hour=12, minute=0, sec=0):
        self.date = datetime.datetime(year, month, day, hour, minute, sec)

    def mktime(self):
        return time.mktime(self.date.timetuple())


t1 = Time(2013, 1, 3, 15)
t2 = Time(2013, 1, 3, 1)

def is_after(time1, time2):
    return time1.mktime() > time2.mktime()

#print is_after(t1, t2)


#Exercise 16.03

class Time(object):
    """ represents the time of day.
    attributes: hour, minute, second"""

time = Time()
time.hour = 11
time.minute = 59
time.second = 30


def increment(time, seconds):
    print ("Original time was: %.2d:%.2d:%.2d"
          % (time.hour, time.minute, time.second))

    time.second += seconds
    if time.second > 59:
        quotient, remainder = divmod(time.second, 60)
        time.minute += quotient
        time.second = remainder
    if time.minute > 59:
        quotient, remainder = divmod(time.minute, 60)
        time.hour += quotient
        time.minute = remainder
    if time.hour > 12:
        time.hour -= 12

    print "Plus %g seconds" % (seconds)
    print "New time is: %.2d:%.2d:%.2d" % (time.hour, time.minute, time.second)

#increment(time, 300)


#Exercise 16.04

import copy


class Time(object):
    """ represents the time of day.
    attributes: hour, minute, second"""

time = Time()
time.hour = 11
time.minute = 59
time.second = 30


def increment(time, seconds):
    print ("Original time was: %.2d:%.2d:%.2d"
          % (time.hour, time.minute, time.second))

    new_time = copy.deepcopy(time)
    new_time.second += seconds
    if new_time.second > 59:
        quotient, remainder = divmod(new_time.second, 60)
        new_time.minute += quotient
        new_time.second = remainder
    if new_time.minute > 59:
        quotient, remainder = divmod(new_time.minute, 60)
        new_time.hour += quotient
        new_time.minute = remainder
    if new_time.hour > 12:
        new_time.hour -= 12

    print "Plus %g seconds" % (seconds)
    print ("New time is: %.2d:%.2d:%.2d"
          % (new_time.hour, new_time.minute, new_time.second))
    print "memory id of object 'time': ", id(time)
    print "memory id of object 'new_time': ", id(new_time)

#increment(time, 300)

#Exercise 16.05

import copy


class Time(object):
    """ represents the time of day.
    attributes: hour, minute, second"""

time = Time()
time.hour = 11
time.minute = 59
time.second = 30


def time_to_int(time):
    minutes = time.hour * 60 + time.minute
    seconds = minutes * 60 + time.second
    return seconds


def int_to_time(seconds):
    new_time = Time()
    minutes, new_time.second = divmod(seconds, 60)
    time.hour, time.minute = divmod(minutes, 60)
    return time


def increment(time, seconds):
    new_time = copy.deepcopy(time)
    new_time = time_to_int(new_time) + seconds
    new_time = int_to_time(new_time)
    print ("New time is: %.2d:%.2d:%.2d"
          % (new_time.hour, new_time.minute, new_time.second))

#increment(time, 300)

#Exercise 16.06

class Time(object):
    """ represents the time of day.
    attributes: hour, minute, second"""

time = Time()
time.hour = 3
time.minute = 0
time.second = 0


def time_to_int(time):
    minutes = time.hour * 60 + time.minute
    seconds = minutes * 60 + time.second
    return seconds


def int_to_time(seconds):
    new_time = Time()
    minutes, new_time.second = divmod(seconds, 60)
    time.hour, time.minute = divmod(minutes, 60)
    return time


def mul_time(time, multicand):
    time_int = time_to_int(time) * multicand
    new_time = int_to_time(time_int)
    if new_time.hour > 12:
        new_time.hour = new_time.hour % 12
#    print ("New time is: %.2d:%.2d:%.2d"
#    % (new_time.hour, new_time.minute, new_time.second))
    return new_time

# mul_time(time, 2)


def race_stats(time, distance):
    print ("The finish time was %.2d:%.2d:%.2d"
          % (time.hour, time.minute, time.second))
    print "The distance was %d miles" % (distance)

    average = mul_time(time, (1.0 / distance))

    print ("The average is: %.2d:%.2d:%.2d per mile"
          % (average.hour, average.minute, average.second))

#race_stats(time, 3)


#Exercise 16.07

import datetime

rules = {0: "Monday",
         1: "Tuesday",
         2: "Wednesday",
         3: "Thursday",
         4: "Friday",
         5: "Saturday",
         6: "Sunday"}


class Time(object):
    now = datetime.datetime.now()

    def __init__(self, year=1, month=1, day=1, hour=0, minute=0, second=0):
        self.date = datetime.datetime(year, month, day, hour, minute, second)

today = Time().now
birthday = Time(1953, 5, 24).date


def day_of_week():
    return "1) Today is %s" % rules[today.weekday()]


def birthday_stats(birthday):
    age = today.year - birthday.year
    if (birthday.month == today.month) and (birthday.day <= today.day):
        pass
    elif birthday.month < today.month:
        pass
    else:
        age -= 1

    birthday_ = Time(today.year, birthday.month, birthday.day).date
    till_birthday = str(birthday_ - today).split()

    if len(till_birthday) > 1:
        days = int(till_birthday[0])
        time = till_birthday[2].split(":")
    else:
        days = 365
        time = till_birthday[0].split(":")

    hours = time[0]
    mins = time[1]
    secs = time[2][:2]

    if (days < 0) and (days != 365):
        days = 365 + days
    elif (days == 365):
        days = 0
    else:
        days = abs(days)

    print ("2) You are %s years old; %sd:%sh:%sm:%ss until your next birthday."
    % (age, days, hours, mins, secs))

#print day_of_week()
#birthday_stats(birthday)


#Exercise 17.01


class Time(object):
    def time_to_int(self):
        minutes = time.hour * 60 + time.minute
        seconds = minutes * 60 + time.second
        return seconds

time = Time()
time.hour = 11
time.minute = 59
time.second = 30

#print time.time_to_int()

#Exercise 17.02


class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def print_point(self):
        print "x =", self.x, ",",
        print "y =", self.y

point = Point()
#point.print_point()

point = Point(10)
#point.print_point()

point = Point(20, 30)
#point.print_point()

#Exercise 17.03

class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return '(%d, %d)' % (self.x, self.y)

point = Point()
#print point

point = Point(10)
#print point

point = Point(10, 15)
#print point

#Exercise 17.04

class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return '(%d, %d)' % (self.x, self.y)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Point(x, y)

point1 = Point(1, 3)
point2 = Point(4, 5)

#print point1 + point2

#Exercise 17.05

class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        point_ = Point()
        if isinstance(other, Point):
            point_.x += self.x + other.x
            point_.y += self.y + other.y
            return point_
        elif type(other) == tuple:
            point_.x += self.x + other[0]
            point_.y += self.y + other[1]
        return point_

    def __radd__(self, other):
        return self.__add__(other)

    def __str__(self):
        return "(%s, %s)" % (self.x, self.y)

point1 = Point(1, 6)
point2 = (5, 2)
point3 = point1 + point2
point4 = point2 + point1
#print point3, point4


#Exercise 17.06

class Time(object):
    """Represents the time of day.

    attributes: hour, minute, second
    """
    def __init__(self, hour=0, minute=0, second=0):
        minutes = hour * 60 + minute
        self.seconds = minutes * 60 + second

    def __str__(self):
        minutes, second = divmod(self.seconds, 60)
        hour, minute = divmod(minutes, 60)
        return '%.2d:%.2d:%.2d' % (hour, minute, second)

    def print_time(self):
        print str(self)

    def time_to_int(self):
        """Computes the number of seconds since midnight."""
        return self.seconds

    def is_after(self, other):
        """Returns True if t1 is after t2; false otherwise."""
        return self.seconds > other.seconds

    def __add__(self, other):
        """Adds two Time objects or a Time object and a number.

        other: Time object or number of seconds
        """
        if isinstance(other, Time):
            return self.add_time(other)
        else:
            return self.increment(other)

    def __radd__(self, other):
        """Adds two Time objects or a Time object and a number."""
        return self.__add__(other)

    def add_time(self, other):
        """Adds two time objects."""
        assert self.is_valid() and other.is_valid()
        seconds = self.seconds + other.seconds
        return int_to_time(seconds)

    def increment(self, seconds):
        """Returns a new Time that is the sum of this time and seconds."""
        seconds += self.seconds
        return int_to_time(seconds)

    def is_valid(self):
        """Checks whether a Time object satisfies the invariants."""
        return self.seconds >= 0 and self.seconds < 24*60*60


def int_to_time(seconds):
    """Makes a new Time object.

    seconds: int seconds since midnight.
    """
    return Time(0, 0, seconds)


def main():
    start = Time(9, 45, 00)
    start.print_time()

    end = start.increment(1337)
    end.print_time()

    print 'Is end after start?',
    print end.is_after(start)

    print 'Using __str__'
    print start, end

    start = Time(9, 45)
    duration = Time(1, 35)
    print start + duration
    print start + 1337
    print 1337 + start

    print 'Example of polymorphism'
    t1 = Time(7, 43)
    t2 = Time(7, 41)
    t3 = Time(7, 37)
    total = sum([t1, t2, t3])
    print total


#if __name__ == '__main__':
#    main()

#Exercise 17.07

class Kangaroo(object):
    """a Kangaroo is a marsupial"""

    def __init__(self, contents=[]):

        self.pouch_contents = contents

    def __init__(self, contents=None):

        if contents == None:
            contents = []
        self.pouch_contents = contents

    def __str__(self):

        t = [ object.__str__(self) + ' with pouch contents:' ]
        for obj in self.pouch_contents:
            s = '    ' + object.__str__(obj)
            t.append(s)
        return '\n'.join(t)

    def put_in_pouch(self, item):

        self.pouch_contents.append(item)

kanga = Kangaroo()
roo = Kangaroo()
kanga.put_in_pouch('wallet')
kanga.put_in_pouch('car keys')
kanga.put_in_pouch(roo)

#print kanga
#print ''

#print roo

#Exercise 17.08








#Exercise 18.01

class Time(object):
    def __init__(self, hour=0, minute=0):
        self.hour = hour
        self.minute = minute

    def __lt__(self, other):
      return (self.hour, self.minute) < (other.hour, other.minute)

    def __gt__(self, other):
      return (self.hour, self.minute) > (other.hour, other.minute)

    def __eq__(self, other):
      return (self.hour, self.minute) == (other.hour, other.minute)

    def __repr__(self):
      return '{}'.format((self.hour, self.minute))

a = Time(hour=3, minute=31)
b = Time(hour=4, minute=30)

#print(a < b)

#Exercise 18.02
import random


class Card(object):
    """Represents a standard playing card.

    Attributes:
      suit: integer 0-3
      rank: integer 1-13
    """

    suit_names = ["Clubs", "Diamonds", "Hearts", "Spades"]
    rank_names = [None, "Ace", "2", "3", "4", "5", "6", "7",
              "8", "9", "10", "Jack", "Queen", "King"]

    def __init__(self, suit=0, rank=2):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        """Returns a human-readable string representation."""
        return '%s of %s' % (Card.rank_names[self.rank],
                             Card.suit_names[self.suit])

    def __cmp__(self, other):
        """Compares this card to other, first by suit, then rank.

        Returns a positive number if this > other; negative if other > this;
        and 0 if they are equivalent.
        """
        t1 = self.suit, self.rank
        t2 = other.suit, other.rank
        return cmp(t1, t2)


class Deck(object):
    """Represents a deck of cards.

    Attributes:
      cards: list of Card objects.
    """

    def __init__(self):
        self.cards = []
        for suit in range(4):
            for rank in range(1, 14):
                card = Card(suit, rank)
                self.cards.append(card)

    def __str__(self):
        res = []
        for card in self.cards:
            res.append(str(card))
        return '\n'.join(res)

    def add_card(self, card):
        """Adds a card to the deck."""
        self.cards.append(card)

    def remove_card(self, card):
        """Removes a card from the deck."""
        self.cards.remove(card)

    def pop_card(self, i=-1):
        return self.cards.pop(i)

    def shuffle(self):
        """Shuffles the cards in this deck."""
        random.shuffle(self.cards)

    def sort(self):
        """Sorts the cards in ascending order."""
        self.cards.sort()

    def move_cards(self, hand, num):
        for i in range(num):
            hand.add_card(self.pop_card())


class Hand(Deck):
    """Represents a hand of playing cards."""

    def __init__(self, label=''):
        self.cards = []
        self.label = label


def find_defining_class(obj, method_name):
    for ty in type(obj).mro():
        if method_name in ty.__dict__:
            return ty
    return None


if __name__ == '__main__':
    deck = Deck()
    deck.shuffle()

    hand = Hand()
   # print find_defining_class(hand, 'shuffle')

    deck.move_cards(hand, 5)
    hand.sort()
    #print hand
#Exercise 18.03





#Exercise 18.04




#Exercise 18.05




#Exercise 18.06


class PokerHand(Hand):

    def suit_hist(self):
        """Builds a histogram of the suits that appear in the hand.

        Stores the result in attribute suits.
        """
        self.suits = {}
        for card in self.cards:
            self.suits[card.suit] = self.suits.get(card.suit, 0) + 1

    def has_flush(self):
        """Returns True if the hand has a flush, False otherwise.

        Note that this works correctly for hands with more than 5 cards.
        """
        self.suit_hist()
        for val in self.suits.values():
            if val >= 5:
                return True
        return False


if __name__ == '__main__':
    # make a deck
    deck = Deck()
    deck.shuffle()

    # deal the cards and classify the hands
  #  for i in range(7):
  #      hand = PokerHand()
  #      deck.move_cards(hand, 7)
  #      hand.sort()
   #     print hand
  #      print hand.has_flush()
  #      print ''
