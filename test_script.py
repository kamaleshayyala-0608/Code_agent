import os
from agent_core import LocalCodeAgentEngine

def run_test():
    models = LocalCodeAgentEngine.list_available_models()
    model_to_use = "gemma4:26b"
    if model_to_use not in models:
        print("Required model not found. Start Ollama and run: ollama pull gemma4:26b")
        return
    print(f"Using model: {model_to_use}")
    engine = LocalCodeAgentEngine(model_name=model_to_use)
    
    test_code = """
def get_user_data(username):
    # This is a bad function
    db_conn = sqlite3.connect("users.db")
    cursor = db_conn.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    data = cursor.fetchall()
    
    result = []
    for d in data:
        for i in range(100):
            result.append(d)
    
    return result
"""
    
    print("=================== CODE REVIEW ===================")
    print(engine.review_code(test_code))
    print("\n=================== ANALYSIS ===================")
    print(engine.analyze_code(test_code))
    print("\n=================== DOCUMENTATION ===================")
    print(engine.document_file("test.py", test_code))
    print("\n=================== REFACTOR ===================")
    print(engine.refactor_file("test.py", test_code))


if __name__ == "__main__":
    run_test()

