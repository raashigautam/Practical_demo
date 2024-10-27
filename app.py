from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load the trained model (ensure the model file is in the same directory)
model = joblib.load("diamond_price_model.pkl")  # replace with your model file

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    data = request.form
    try:
        # Convert form inputs into the required format (replace placeholders)
        carat = float(data['carat'])
        cut = data['cut']
        color = data['color']
        clarity = data['clarity']
        depth = float(data['depth'])
        table = float(data['table'])
        x = float(data['x'])
        y = float(data['y'])
        z = float(data['z'])

        # Create a DataFrame for model input
        input_data = pd.DataFrame({
            'carat': [carat],
            'cut': [cut],
            'color': [color],
            'clarity': [clarity],
            'depth': [depth],
            'table': [table],
            'x': [x],
            'y': [y],
            'z': [z]
        })

        # Make a prediction
        prediction = model.predict(input_data)[0]

        return jsonify({'predicted_price': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
