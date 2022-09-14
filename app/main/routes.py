from base64 import b64encode
from io import BytesIO
import cv2
from PIL import Image
from flask import  Response, flash
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed
from werkzeug.exceptions import abort
from wtforms import FileField , SubmitField
from app.main import main_bp
from app.main.camera import Camera
from source.img import image_detect
from source.video_detector import video_detect
from flask import  redirect, url_for
from source.api import *
from source.cal_vol import imagedtct
import sqlite3
import hashlib
import sqlite3 as sql
from flask import render_template,render_template_string
from flask import request
import models as dbHandler

def insertUser(username,password):
    con = sql.connect("C:/Users/User/Desktop/project/app/static/User.db")
    cur = con.cursor()
    cur.execute("INSERT INTO users (username,password) VALUES (?,?)", (username,password))
    con.commit()
    con.close()

def retrieveUsers():
	con = sql.connect("C:/Users/User/Desktop/project/app/static/User.db")
	cur = con.cursor()
	cur.execute("SELECT username, password FROM users")
	users = cur.fetchall()
	con.close()
	return users

def check_password(hashed_password, user_password):
    return hashed_password == hashlib.md5(user_password.encode()).hexdigest()

def validate(username, password):
    con = sqlite3.connect('C:/Users/User/Desktop/project/app/static/User.db')
    completion = False
    with con:
                cur = con.cursor()
                cur.execute("SELECT * FROM Users")
                rows = cur.fetchall()
                for row in rows:
                    dbUser = row[0]
                    dbPass = row[1]
                    if dbUser==username:
                        completion=check_password(dbPass, password)
    return completion


@main_bp.route("/")
def home_page():
    return render_template("index.html")



@main_bp.route('/sign_up', methods=['POST', 'GET'])
def sign_up():
    if request.method=='POST':
        username = request.form['username']
        password = request.form['password']
        dbHandler.insertUser(username, password)
        users = dbHandler.retrieveUsers()
        return render_template('sign_up.html', users=users)
    else:
        return render_template('sign_up.html')

@main_bp.route('/live_detector')
def live_detector():
    return render_template("live_detector.html")
def gen(camera):

    while True:
        frame = camera.get_frame()
        frame_processed = video_detect(frame)
        frame_processed = cv2.imencode('.jpg', frame_processed)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_processed + b'\r\n')

@main_bp.route('/video_feed')
def video_feed():
    return Response(gen(
        Camera()
    ),
        mimetype='multipart/x-mixed-replace; boundary=frame')
def allowed_file(filename):
    ext = filename.split(".")[-1]
    is_good = ext in ["jpg", "jpeg", "png"]
    return is_good


@main_bp.route("/image-detector", methods=["GET", "POST"])
def image_detection():
    return render_template("image_detector.html",
                           form=PhotoMaskForm())


@main_bp.route("/image-processing", methods=["POST"])
def image_processing():
    form = PhotoMaskForm()

    if not form.validate_on_submit():
        flash("An error occurred", "danger")
        abort(Response("Error", 400))

    pil_image = Image.open(form.image.data)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    array_image = image_detect(image)
    rgb_image = cv2.cvtColor(array_image, cv2.COLOR_BGR2RGB)
    image_detected = Image.fromarray(rgb_image, 'RGB')

    with BytesIO() as img_io:
        image_detected.save(img_io, 'PNG')
        img_io.seek(0)
        base64img = "data:image/png;base64," + b64encode(img_io.getvalue()).decode('ascii')
        return base64img


# form
class PhotoMaskForm(FlaskForm):
    image = FileField('Choose image:',
                      validators=[
                          FileAllowed(['jpg', 'jpeg', 'png'], 'The allowed extensions are: .jpg, .jpeg and .png')])

    submit = SubmitField('Estimate')

@main_bp.route("/cal_vol", methods=["GET", "POST"])
def cal_vol():
    return render_template("cal_vol.html",
                           form=PhotoMaskForm())
@main_bp.route("/image-pro", methods=["POST"])
def image_pro():
    form = PhotoMaskForm()

    if not form.validate_on_submit():
        flash("An error occurred", "danger")
        abort(Response("Error", 400))

    pil_image = Image.open(form.image.data)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    array_image = imagedtct(image)
    rgb_image = cv2.cvtColor(array_image, cv2.COLOR_BGR2RGB)
    image_detected = Image.fromarray(rgb_image, 'RGB')

    with BytesIO() as img_io:
        image_detected.save(img_io, 'PNG')
        img_io.seek(0)
        base64img = "data:image/png;base64," + b64encode(img_io.getvalue()).decode('ascii')
        return base64img
@main_bp.route('/poid')
def poid():
    return render_template('poid.html')
@main_bp.route('/nutrition')
def nutrition():
    return render_template('nutrition.html')

@main_bp.route('/about')
def about_page():
    return render_template("about.html")

@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        completion = validate(username, password)
        if completion ==False:
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('secret'))
    return render_template('login.html', error=error)

@main_bp.route('/secret')
def secret():
    return "You have successfully logged in"
