import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

#text_to_analyze = "The product arrived two days late, but the customer service team was so helpful that I'm actually quite happy with the purchase."


test_examples = [
    # --- English Examples ---
    "This is the best purchase I've made all year! Absolutely worth every penny.", # Strong Positive
    "I'm not sure how I feel about it. It works, but it's nothing special.",        # Neutral/Mixed
    "The interface is a nightmare and it crashes every 10 minutes. Avoid it.",      # Strong Negative
    "Oh great, another 'feature' that makes the app slower. Just what I wanted.",   # Sarcastic (Testing Reasoning)
    "The delivery took 5 days and arrived in a blue box.",                         # Neutral/Factual

    # --- Arabic Examples (MSA & Dialect) ---
    "التطبيق رائع جداً وسهل الاستخدام، أنصح الجميع بتحميله.",                        # Positive (MSA)
    "للأسف تجربة سيئة للغاية، المنتج وصل مكسوراً والدعم الفني لا يجيب.",              # Negative (MSA)
    "وصل الشحن اليوم في تمام الساعة الرابعة عصراً.",                                # Neutral (MSA)
    "يا بلاش! جودة ممتازة وسعر رهيب، عاشت ايديكم.",                                # Positive (Egyptian/Gulf Dialect)
    "والله ما يسوى ولا فلس، ضياع وقت وفلوس على الفاضي.",                           # Negative (Levantine/Gulf Dialect)
    "الجهاز لونه أسود وحجمه متوسط."                                               # Neutral (MSA)
]

for i, text_to_analyze in enumerate(test_examples):
    print(f"\n--- Testing Example {i+1} ---")
    print(f"Text: {text_to_analyze}")


    prompt_reasoning = f"""
    Analyze the sentiment of the following text. Provide your response in the following format:
    Sentiment: [Positive/Negative/Neutral]
    Reasoning: [A brief explanation of why]

    Text: {text_to_analyze}
    """


    try:
        response = llm.invoke(prompt_reasoning)
        print("--- Analysis with Reasoning ---")
        print(response.content)
    except Exception as e:
        print("An error occurred:", e)