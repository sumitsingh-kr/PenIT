from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello World!"
if __name__=='__main__':
    app.run(port='80')
