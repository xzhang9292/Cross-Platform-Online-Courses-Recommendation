from django.shortcuts import render
import json
import textdistance
import pandas as pd

    




with open('blog/course_matching_map_out_all_plat.json', 'r') as f:
    matches = json.load(f)

with open('blog/joblabel.json','r') as f2:
    joblabel = json.load(f2)


with open('blog/title.json','r') as f3:
    titles = json.load(f3)

with open('blog/url_all_plat.json','r') as f4:
    url = json.load(f4)
with open('blog/platform_all_plat.json','r') as f5:
    plats = json.load(f5)




def getcourseinfo(title):
    return {'title':title, 'url':url[title], 'plat':plats[title]}
def create_courselist(courses):
    return [getcourseinfo(k) for k in courses]

def home(request):
    courses = matches['0']
    posts = create_courselist(courses)

    context = {
        'posts': posts
    }
    return render(request, 'blog/home.html', context)

def key_input_search(key):
    max_score = 0.0
    index = 0
    l = len(titles)
    for i in range(l):
        if not pd.isna(titles[str(i)]):
            score = textdistance.jaro_winkler(titles[str(i)].lower(),key.lower())
            if score > max_score:
                max_score = score
                index = i
    return joblabel[str(index)]



def search(request):
    if request.method == 'POST':
        key = request.POST.get('textfield', None)

        label = key_input_search(key)
        print('the label {}'.format(label))
        courses = matches[str(label)]
        posts = create_courselist(courses)
        context = {
        'posts': posts
        }
        
    else:
        courses = matches['0']
        posts = create_courselist(courses)
        context = {
        'posts': posts
        }
    return render(request, 'blog/home.html', context)
def about(request):

    return render(request, 'blog/about.html', {'title': 'About'})
