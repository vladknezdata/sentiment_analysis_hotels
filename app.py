from flask import Flask, render_template, request, jsonify, redirect, send_from_directory
import pymongo
import os
from sentiment import sentiment, sb, sf
#from sentimentBoston import sentimentBoston

app = Flask(__name__)
stored_object = {}
output_images = []
#target = 'static/images'
#output_dir = 'static/output'

'''if not os.path.isdir(target):
    os.mkdir(target)
else:
    print("Couldn't create upload directory: {}".format(target))
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
else:
    print("Couldn't create output directory: {}".format(output_dir))'''

"""@app.route("/")
def homepage():
    # store mongo data in global variable
    sentiment.sentiment()
    ###global stored_object
    # fetch all db data and place into api
    ###stored_object = mongoFetchClasses()
    ###return render_template('index.html')
"""
@app.route("/fetch/api")
def fetchapi():
    # return mongoDB as JSON data
    return jsonify(stored_object)

@app.route("/index")
def index():
    #path = '/Users/vlad/Desktop/Columbia/projects/Sentiment-Analysis-for-Hotel-Reviews/'
    #sentiment()
    #sb()
    #sf()
    #output_images = ['static/output/' + img for img in os.listdir('static/output/')]
    return render_template('index.html')#, output_images=output_images)

@app.route("/bos")
def boston():
    return render_template('bos.html')

@app.route("/nyc")
def nyc():
    return render_template('nyc.html')

@app.route("/sanfran")
def sanfran():
    return render_template('sanfran.html')

@app.route('/uploadfiles/saveimages', methods=["POST"])
def saveimages():
    for upload in request.files.getlist("file"):
        print("{} is the filename".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        image_path = 'static/images/' + upload.filename
        # print("Save it to:", destination)
        upload.save(destination)
        file_name_root = upload.filename.split('.')[0]
        # create an output file name 
        file_output_name = file_name_root + '_output.png'
        # append file output name for later
        output_images.append(file_output_name)
        # run ML function
        image, output_image_matrix, classes_detected = maskImage(destination, ('website/static/output/' + file_output_name))
        os.remove(destination)

    # redirect to home page
    return redirect("/uploadfiles")







if __name__ == "__main__":
    app.run(debug=False)