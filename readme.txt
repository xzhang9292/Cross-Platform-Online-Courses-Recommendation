
To Run Website (Mac):
1. go to django project folder
cd Search_Website/django_project/

2. run server with manage.py
python manage.py runserver
Note following message in terminal and find port:
"Starting development server at http://X.X.X.X:8000/"

3. go to local hosted website. In this case: http://localhost:8000/

Note:
Make sure textdistance and pandas has been installed.
To install in terminal:
pip install textdistance
pip install pandas

#######################################
File explanation:

Collect_Data/data_collect_clean.py
Collect Data with API calls and clean the result then save the data in .npy format.
Secret Token has been removed from original code.

Matching_Model/match_model.py
Load data in .npy format. Do Job Clustering, key word extortion and key word matching. The result are saved in json format.
"glove.6B.200d.txt" which used for word embedding has been removed due to its size.

Search_Website/
Contains code for search website written in Django Framework

Merged_Code.ipynb:
Merged all the above codes in a single notebook. It also added user input matching which is written in the backend of search website

########################################
Updates from last presentation:
Added Courses from Udacity and Coursera
Added data source information in ppt and final report

