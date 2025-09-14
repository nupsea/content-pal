from flask import Flask, request, jsonify
from src.modules.workflow.rag import rag
from src.modules.workflow import db
import uuid

app = Flask(__name__)

@app.route('/')
def health():
    return jsonify({"status": "healthy", "service": "content-pal-api"})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    query = data['query'] 
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    conversation_id = str(uuid.uuid4())
    
    recommendations = rag(query)
    db.save_conversation(conversation_id, query, recommendations)
    
    # Add conversation_id to response for feedback
    recommendations["conversation_id"] = conversation_id
    return jsonify(recommendations)


@app.route("/feedback", methods=["POST"])
def handle_feedback():
    data = request.json
    conversation_id = data["conversation_id"]
    feedback = data["feedback"]

    if not conversation_id or feedback not in [1, -1]:
        return jsonify({"error": "Invalid input"}), 400

    db.save_feedback(
        conversation_id=conversation_id,
        feedback=feedback,
    )

    result = {
        "message": f"Feedback received for conversation {conversation_id}: {feedback}"
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

