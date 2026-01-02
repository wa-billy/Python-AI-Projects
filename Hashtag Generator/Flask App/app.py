import os
import base64
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from main import encode_image

load_dotenv()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    base64_image = None
    if request.method == 'POST':
        image = request.files['image']
        image.save("uploaded_image.jpg")

        image_base64 = encode_image("uploaded_image.jpg")

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv("OPENAI_KEY")}"
        }

        payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a hashtag generation model. When you get an image as input, your response should always contain exactly 30 hashtags separated by commas."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Provide the hashtags for this image:"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }
        
        response = requests.post(url='https://api.openai.com/v1/chat/completions', headers=headers, json=payload)

        hashtags = response.json()['choices'][0]['message']['content'].split(',')
        hashtags = [h.strip() for h in hashtags]

        return render_template('index.html', hashtags=hashtags, base64_image=base64_image)
        

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
