import gradio as gr
import random
import time
import requests

def chat_completion(usr_message):
    #prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
    api_url = "http://127.0.0.1:5000/api/ragllama2"
    body = {
            "user_message": usr_message
            }
    response = requests.post(api_url, json=body)
    formatted_response = response.json()
    print(formatted_response['Response'])
    return formatted_response

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        #bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        bot_response = chat_completion(message)
        msgDisplayedOnBot = bot_response['Response']
        bot_message = msgDisplayedOnBot
        chat_history.append((message, bot_message))
        time.sleep(2)
        print("Message: ", message)
        #print("Chat History: ",chat_history)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
