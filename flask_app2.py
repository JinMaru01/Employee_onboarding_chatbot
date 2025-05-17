from flask import Flask, render_template, request, jsonify

app = Flask(__name__, static_folder='static')

@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
