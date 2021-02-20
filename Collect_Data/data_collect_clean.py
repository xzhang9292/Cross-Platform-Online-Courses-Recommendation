import requests
from urllib import request
import urllib
import json, codecs
import requests
import pymongo


def getCourseList():
    url = "https://courses.edx.org/api/courses/v1/courses/?page=1"
    courseList = []
    while True:
        
        response = request.urlopen(url)
        data = json.loads(response.read())

        for course in data['results']:
            courseList.append(course['id'])

        url = data['pagination']['next']
        if url == None:
            break
    return courseList

def parseCourse(rawdata):
    baseUrl = "https://www.edx.org"
    course = []
    title = rawdata['title']
    description = rawdata['description']
    what_to_learn = rawdata['what_you_will_learn']
    course = [title,description,what_to_learn]
    return course

def getResponse(url):
    headers = {"Authorization": auth}
    req = requests.get(url, headers = headers)
    data = req.json()
    
    return data
def getCourseDetail(courseList):
    baseUrl = "https://www.edx.org/api/catalog/v2/courses/"
    courseDetail = []
    for uri in courseList:
        url = baseUrl + uri
        try:
            response = getResponse(url)
        except:
            print(url)
            continue
        if type(response) is dict:
            #course = parseCourse(response)
            courseDetail.append(response)

    return courseDetail

import re
def clean(text):
    # Special characters
    text = re.sub(r"\x89Û_", "", text)
    text = re.sub(r"\x89ÛÒ", "", text)
    text = re.sub(r"\x89ÛÓ", "", text)
    text = re.sub(r"\x89ÛÏWhen", "When", text)
    text = re.sub(r"\x89ÛÏ", "", text)
    text = re.sub(r"China\x89Ûªs", "China's", text)
    text = re.sub(r"let\x89Ûªs", "let's", text)
    text = re.sub(r"\x89Û÷", "", text)
    text = re.sub(r"\x89Ûª", "", text)
    text = re.sub(r"\x89Û\x9d", "", text)
    text = re.sub(r"å_", "", text)
    text = re.sub(r"\x89Û¢", "", text)
    text = re.sub(r"\x89Û¢åÊ", "", text)
    text = re.sub(r"fromåÊwounds", "from wounds", text)
    text = re.sub(r"åÊ", "", text)
    text = re.sub(r"åÈ", "", text)
    text = re.sub(r"JapÌ_n", "Japan", text)    
    text = re.sub(r"Ì©", "e", text)
    text = re.sub(r"å¨", "", text)
    text = re.sub(r"SuruÌ¤", "Suruc", text)
    text = re.sub(r"åÇ", "", text)
    text = re.sub(r"å£3million", "3 million", text)
    text = re.sub(r"åÀ", "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"\t", "", text)
    text = re.sub(r"[ ]+", " ", text)

    # Character entity references
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&amp;", "&", text)
    
    # Urls
    text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", text)
    # Words with punctuations and special characters
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        text = text.replace(p, f' {p} ')
    #remove html tag
    html = re.compile(r'<.*?>')
    text = html.sub(r'',text)
    return text
    
if __name__ == '__main__':
	courseList = getCourseList()
	#secret token has been removed
	auth = ""
	courseDetail = getCourseDetail(courseList)
	courseDetail_array = np.array(courseDetail)
	np.save('coursesdetail.npy', courseDetail_array)
	course_data1 = np.load('coursesdetail.npy')

	#get data from MOOcer 2.0
	client = pymongo.MongoClient('mongodb://admin:a123456@ds339648.mlab.com:39648/moocer')
	db = client.moocer
	course_data2 = []
	for x in db.courses.find():
	    course_data2.append(x)

	# merge two data source
	coursesinfo = []
	for c in course_data1:
	    
	    if (not c['title'] is None) and (not c['description'] is None) and (len(c['description']) > 0) and (not c['what_you_will_learn'] is None) and (len(c['what_you_will_learn']) > 0 ):
	        coursesinfo.append([c['title'],c['description'],c['what_you_will_learn'],c['course_about_uri'],'edX'])
	for s in course_data2:
	    if (not s['title'] is None) and (not s['summary'] is None) and (len(s['summary']) > 0) and (len(s['shortSummary']) > 0):
	        coursesinfo.append([s['title'],s['summary'],s['shortSummary'],s['url'],s['platform']])
	    
	coursesinfo = np.array(coursesinfo)
	for row in coursesinfo:
	    row[1] = clean(row[1])
	    row[2] = clean(row[2])
	np.save('coursesinfo.npy', coursesinfo)