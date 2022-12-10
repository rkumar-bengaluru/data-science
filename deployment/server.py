from flask import Flask,request,jsonify
import joblib 
import pandas as pd

# create flask application
app = Flask(__name__)
# connect post to predict function
# curl -H "Content-Type: application/json" -X POST http://localhost:5000/predict -d "[{\"TV\":230.1,\"radio\":37.8,\"newspaper\":69.2}]"
@app.route('/predict',methods=['POST'])
def predict():
    # get the json request
    feature_data = request.json
    print('hello world')
    print(feature_data)
    # extract json to pandas dataframe
    df = pd.DataFrame(feature_data)
    df = df.reindex(columns=col_names)
    # predict
    prediction = list(model.predict(df))
    # return prediction
    return jsonify({'prediction' : str(prediction)})


# load model and column name
if __name__ == '__main__':
    model = joblib.load('final_model.pkl')
    col_names = joblib.load('col_names.pkl')
    app.run(debug=True)