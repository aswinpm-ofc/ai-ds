from flask import Flask, render_template, request, jsonify
from nlp.suggestion_engine import SuggestionEngine

app = Flask(__name__)
engine = SuggestionEngine("corpus/corpus.txt")

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/suggest")
def suggest():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    return jsonify({"suggestions": engine.suggest(text, limit=5)})

if __name__ == "__main__":
    app.run(debug=True)
