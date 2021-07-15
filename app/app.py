from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy

app = Flask(__name__)
app = Flask(__name__, template_folder='template')

dic = {0. : 'Cat', 1. : 'Dog'}

model = load_model('cnn_model.h5')


def predict_label(img_path):
	i = image.load_img(img_path, target_size=(100,100))
	i = image.img_to_array(i)/255.0
	i = i.reshape(100,100,3)
	i   = numpy.expand_dims(i, axis=0)
	p = model.predict(i)

	return dic[p[1]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "This is End-to-End Image Classification Web App"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

		return render_template("index.html", prediction = p , img_path = img.filename)


if __name__ =='__main__':
	app.debug = True
	app.run(debug = True)