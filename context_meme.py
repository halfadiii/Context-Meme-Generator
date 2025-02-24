import os
import logging
import requests
import random
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from groq import Groq
import sys

class MemeGenerator:
    """
    A class to generate memes based on chat context using LLaMA 3-70B API via Groq and Imgflip API.
    """

    def __init__(self, api_key: str, imgflip_username: str, imgflip_password: str):
        """
        Initialize the MemeGenerator with Groq API credentials and Imgflip credentials.
        """
        os.environ["GROQ_API_KEY"] = api_key
        self.client = Groq()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        self.imgflip_username = imgflip_username
        self.imgflip_password = imgflip_password

    def extract_topic_from_chat(self, chat_context: str) -> str:
        """
        Uses LLaMA 3-70B to analyze chat and determine the meme topic.
        """
        messages = [
            {
                "role": "system",
                "content": "Analyze the given conversation and return only a single-word topic "
                           "(e.g., 'coding', 'gym', 'AI', 'exams', 'sleep', etc.) that best represents the discussion."
            },
            {
                "role": "user",
                "content": chat_context
            }
        ]

        try:
            completion = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=0.7,
                max_completion_tokens=5,
                top_p=1,
                stream=True,
                stop=None,
            )

            topic = ""
            for chunk in completion:
                topic += chunk.choices[0].delta.content or ""

            return topic.strip().lower()

        except Exception as e:
            self.logger.error(f"Error extracting topic: {str(e)}")
            return "funny" #default

    def fetch_meme_template(self):
        """
        Fetches the latest meme templates from Imgflip API.
        """
        response = requests.get("https://api.imgflip.com/get_memes")
        
        if response.status_code == 200:
            memes = response.json()["data"]["memes"]
            selected_meme = random.choice(memes)  # Random meme template
            return selected_meme["id"], selected_meme["url"], selected_meme["name"]
        else:
            return None, None, None #fail
        
    def generate_meme_caption(self, topic: str, meme_name: str) -> str:
        """
        Uses LLaMA 3-70B to generate a contextual meme caption based on the topic and specific meme.
        Ensures AI does not include explanations.
        """
        messages = [
            {
                "role": "system",
                "content": f"You are a meme expert. Generate ONLY a short, funny meme caption for the topic '{topic}'. "
                           f"DO NOT add any explanations, introductions, or extra text. Just output the caption."
            }
        ]

        try:
            completion = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=1.0,
                max_completion_tokens=15,
                top_p=1,
                stream=True,
                stop=None,
            )

            meme_caption = ""
            for chunk in completion:
                meme_caption += chunk.choices[0].delta.content or ""
            return meme_caption.strip().replace('"', '')

        except Exception as e:
            self.logger.error(f"Error generating meme caption: {str(e)}")
            return "Me debugging at 3 AM..."  #default caption

    def create_meme(self, meme_id: str, text: str):
        """
        Uses Imgflip API to overlay text on a meme template.
        """
        params = {
            "template_id": meme_id,
            "username": self.imgflip_username,
            "password": self.imgflip_password,
            "text0": text,
            "text1": "" #empty text (testing)
        }

        response = requests.post("https://api.imgflip.com/caption_image", data=params)
        
        if response.status_code == 200 and response.json()["success"]:
            meme_url = response.json()["data"]["url"]
            print(f"\n✅ Meme Generated: {meme_url}")
        else:
            print("❌ Error generating meme on Imgflip.")

    def generate_meme_from_chat(self, chat_context: str):
        """
        Processes chat history and generates a meme based on it.
        """
        topic = self.extract_topic_from_chat(chat_context)
        meme_id, meme_image_url, meme_name = self.fetch_meme_template()
        
        if not meme_id:
            print("❌ Error fetching meme template.")
            return

        meme_text = self.generate_meme_caption(topic, meme_name)

        print(f"🖼 Meme Topic: {topic}")
        print(f"📝 Generated Meme Text: {meme_text}")
        print(f"🔗 Using Meme Template: {meme_name} ({meme_image_url})")

        self.create_meme(meme_id, meme_text)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    """
    Example usage of the MemeGenerator class.
    """
    api_key = "YOUR_YOUR_OWN_GODDAMN_API"
    imgflip_username = "I_AM_GUESSING_YOU'D_HAVE_YOUR_OWN_USERNAME"
    imgflip_password = "YOU_SERIOUSLY_DONT_REMEMBER_YOUR_PASSWORD?" 

    meme_generator = MemeGenerator(api_key, imgflip_username, imgflip_password)

    chat_history = "I think my Girlfriend cheated on me bro"
    meme_generator.generate_meme_from_chat(chat_history)


if __name__ == "__main__":
    main()
