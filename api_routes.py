from flask import Blueprint, request, jsonify
from api_logic import LLM_Models

api_bp = Blueprint('api', __name__)
llmModels = LLM_Models()


@api_bp.route('/ragllama2', methods=['POST'])
def llama2():
    data = request.get_json()
    if 'user_message' not in data:
        return jsonify({'error': 'Please provide a valid query'}), 400
    user_message = data['user_message']

    try:
        result = llmModels.llama2(user_message)
        #response_data = {'result': result}
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
