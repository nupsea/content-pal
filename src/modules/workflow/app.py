from flask import Flask, request, jsonify
from src.modules.workflow.rag import rag

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    query = data['query'] 
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    recommendations = rag(query)
    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)

