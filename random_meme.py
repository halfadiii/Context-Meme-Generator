import os
import logging
import requests
import random
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import praw
from groq import Groq

class MemeGenerator:
    """
    A class to generate memes based on chat context using LLaMA 3-70B API via Groq and memes from Reddit.
    """

    def __init__(self, api_key: str, reddit_client_id: str, reddit_client_secret: str, reddit_user_agent: str):
        """
        Initialize the MemeGenerator with Groq API and Reddit API credentials.
        """
        os.environ["GROQ_API_KEY"] = api_key
        self.client = Groq()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Reddit API client
        self.reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )

    def extract_topic_from_chat(self, chat_context: str) -> str:
        """
        Uses LLaMA 3-70B to analyze chat and determine the meme topic.
        """
        messages = [
            {
                "role": "system",
                "content": "Analyze the given conversation and return only a single-word topic (e.g., 'coding', 'gym', 'AI', 'exams', 'sleep', etc.) that best represents the discussion."
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
            return "funny"  # Default topic 

    def generate_meme_caption(self, topic: str) -> str:
        """
        Uses LLaMA 3-70B to generate a witty meme caption.
        """
        messages = [
            {
                "role": "system",
                "content": f"Generate a funny meme caption about {topic}. Keep it short and witty, under 10 words."
            }
        ]

        try:
            completion = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=1.0,
                max_completion_tokens=20,
                top_p=1,
                stream=True,
                stop=None,
            )

            meme_caption = ""
            for chunk in completion:
                meme_caption += chunk.choices[0].delta.content or ""

            return meme_caption.strip()

        except Exception as e:
            self.logger.error(f"Error generating meme caption: {str(e)}")
            return "When life gives you errors, debug them!"  # Default caption 

    def fetch_meme_from_reddit(self) -> str:
        """
        Fetches a random meme image URL from r/memes subreddit.
        """
        try:
            subreddit = self.reddit.subreddit("memes")
            memes = [post for post in subreddit.hot(limit=50) if not post.stickied and post.url.endswith(("jpg", "png"))]

            if memes:
                meme_post = random.choice(memes)
                return meme_post.url
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error fetching meme from Reddit: {str(e)}")
            return None

    def overlay_text_on_image(self, image_url: str, text: str):
        """
        Downloads a meme image and overlays AI-generated text on it.
        """
        try:
            response = requests.get(image_url)
            if response.status_code != 200:
                print("Error fetching meme image.")
                return

            img = Image.open(BytesIO(response.content))
            draw = ImageDraw.Draw(img)

            # default font
            font = ImageFont.truetype("arial.ttf", size=40)

            #text placement
            width, height = img.size
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]


            x = (width - text_width) // 2
            y = height - 100  #caption position

            #outline
            draw.text((x - 2, y - 2), text, font=font, fill="black")
            draw.text((x + 2, y + 2), text, font=font, fill="black")
            draw.text((x, y), text, font=font, fill="white")

            # Save
            img.save("generated_meme.jpg")
            img.show()
            print("\nMeme generated successfully!")

        except Exception as e:
            self.logger.error(f"Error overlaying text on image: {str(e)}")

    def generate_meme_from_chat(self, chat_context: str):
        """
        Processes chat history and generates a meme based on it.
        """
        topic = self.extract_topic_from_chat(chat_context)
        meme_text = self.generate_meme_caption(topic)
        meme_url = self.fetch_meme_from_reddit()

        if meme_url:
            print(f"Reddit Meme URL: {meme_url}")
            self.overlay_text_on_image(meme_url, meme_text)
        else:
            print("Failed to fetch meme from Reddit.")


def main():
    """
    Example usage of the MemeGenerator class.
    """
    api_key = "YOU_YOUR_OWN_GODDAMN_API"
    reddit_client_id = "YOU_YOUR_OWN_GODDAMN_CLIENT_ID"
    reddit_client_secret = "YOU_YOUR_OWN_GODDAMN_SECRET_ID"
    reddit_user_agent = "YOU_YOUR_OWN_GODDAMN_USER_AGENT"

    meme_generator = MemeGenerator(api_key, reddit_client_id, reddit_client_secret, reddit_user_agent)

    chat_history = "Bro, I pulled an all-nighter debugging and found out the issue was a missing semicolon!"
    meme_generator.generate_meme_from_chat(chat_history)


if __name__ == "__main__":
    main()
