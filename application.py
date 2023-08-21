from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
import joblib
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    sex = 1 if request.form['sex'] == 'female' else 0
    fare = float(request.form['fare'])

    # Make a prediction using the trained model
    prediction = model.predict([[age, sex, fare]])

    result = "Survived" if prediction[0] == 1 else "Not Survived"
    return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)
