from flask import Flask, json, request, jsonify, send_file
from flask_cors import CORS
from flask_marshmallow import Marshmallow #ModuleNotFoundError: No module named 'flask_marshmallow' = pip install flask-marshmallow https://pypi.org/project/flask-marshmallow/
import PIL
from PIL import Image as pil_image
import io
from model import db, Image
import os
import urllib.request
from werkzeug.utils import secure_filename #pip install Werkzeug
from io import BytesIO
from flask_sqlalchemy import SQLAlchemy
import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import base64


#-----------------------------------------------------------------------------------------------
import os
HOME = os.getcwd()
print(HOME)



#from IPython import display
#display.clear_output()

import sys
#sys.path.append("C:\Users\MSII\Desktop\AI Detecting Counting Image\backend\ByteTrack") 


from dataclasses import dataclass





#-----------------------------------------------------------------------------------------------


app = Flask(__name__)
CORS(app, supports_credentials=True)

#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flaskdb.db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///image_database.db'
db = SQLAlchemy(app)
# Databse configuration mysql                             Username:password@hostname/databasename
#app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:''@localhost/flaskreact'

#db.init_app(app)
  
with app.app_context():
    db.create_all()
 
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
ma=Marshmallow(app)
 
class ImageSchema(ma.Schema):
    class Meta:
        fields = ('id','title')
         
image_schema = ImageSchema(many=True)


class Image(db.Model):
    __tablename__ = "image_entries"
    _id = db.Column(db.Integer, primary_key=True, unique=True)
    competitor_name = db.Column(db.String(120), index=True)
    date = db.Column(db.String(120), index=True)
    image_path = db.Column(db.LargeBinary)

@app.route("/")
def hello_world():
    return "Hello, World!"



def process_image(image_path):
    image = Image.query.get(image_path)
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


    reader = easyocr.Reader(['en'], gpu=False)

    text_ = reader.readtext(img)

    listS = []

    threshold = 0.25

    for t in text_:
        bbox, text, score = t

        if score > threshold:
            cv2.rectangle(img, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
            cv2.putText(img, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX, 2.5, (255, 0, 0), 2)
            listS.append(text)

    for i in listS:
        print(i)

    with open('DataOutput.csv', 'a+') as f:
        for data in listS:
            f.write((str(data) + '\n'))
        f.write('\n')

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


@app.route('/upload', methods=['POST'])
def upload_file():
    #data = jsonify(request.get_json())
    #process_image('2')
    #print(data.id)

    # Assuming 'image_id' is provided in the request payload
    image_id = request.json.get('image_id')

    if image_id is None:
        return jsonify({
            "message": 'Image ID not provided',
            "status": 'failed'
        }), 400


    # Fetch the first row from the "image_entries" table
    image_entry = Image.query.first()

    if image_entry:
        # Retrieve image data from the "image_path" column
        image_data = image_entry.image_path

        # Save the image to the local folder
        image_path = f'static/uploads/image_{image_id}.jpg'  # Assuming it's a JPEG image
        with open(image_path, 'wb') as image_file:
            image_file.write(image_data)

        # Call process_image with the saved image path
        process_image(image_path)

        return jsonify({
            "message": 'Image processing initiated',
            "status": 'success'
        }), 200
    else:
        return jsonify({
            "message": 'No images found in the database',
            "status": 'failed'
        }), 404


    return jsonify(request.json)
    
   
   






     
@app.route('/images',methods =['GET'])
def images():
    # all_images = Image.query.all()
    # results = image_schema.dump(all_images)
    # return jsonify(results)

    all_images = Image.query.all()
    image_list = []

    for image in all_images:
        image_list.append({
            'id': image._id,
            'competitor_name': image.competitor_name,
            'date': image.date,
            'image_path': f'/get_image/{image._id}'  # Endpoint to retrieve the image
        })

    return jsonify(image_list)



@app.route('/get_image/<int:image_id>', methods=['GET'])
def get_image(image_id):
    image = Image.query.get(image_id)

    if image:
        return send_file(
            io.BytesIO(image.image_path),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'image_{image_id}.jpeg'
        )
    else:
        return jsonify({
            "message": 'Image not found',
            "status": 'failed'
        }), 404


# @app.route('/clear', methods=['GET'])
# def clear_images():
#     try:
#         db.session.query(Image).delete()
#         db.session.commit()
#         return jsonify({
#             "message": 'All images deleted successfully',
#             "status": 'success'
#         }), 200
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({
#             "message": 'Failed to delete images',
#             "status": 'failed',
#             "error": str(e)
#         }), 500
    


