import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)

def analyze_feedback():
    feedback_data = {
        'Review_ID':,
        'Product': ['Laptop Charger', 'Wireless Mouse', 'Keyboard', 'USB-C Cable', 'Bluetooth Speaker'],
        'Customer_Comment': [
            "The charger overheats within 10 minutes. Absolute waste of money, dangerous!",
            "Arrived quickly and works flawlessly. Super quiet clicking, love the design.",
            "It is an okay keyboard, but the spacebar keys feel slightly sticky.",
            "Completely dead on arrival. Did not charge my phone at all. Sending it back.",
            "Incredible battery life and booming sound quality! Highly recommend this speaker."
        ],
        'Rating': [1, 5, 3, 1, 5]
    }
    
    df = pd.DataFrame(feedback_data)
    sia = SentimentIntensityAnalyzer()
    
    sentiments = []
    for comment in df['Customer_Comment']:
        score = sia.polarity_scores(comment)['compound']
        if score >= 0.05:
            sentiments.append('Positive')
        elif score <= -0.05:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')
            
    df['AI_Sentiment_Tag'] = sentiments
    
    print("--- RAW ALIGNED DATASET ---")
    print(df[['Product', 'Rating', 'AI_Sentiment_Tag']])
    
    print("\n--- KPI METRIC: AVERAGE RATING BY AI SENTIMENT CLASS ---")
    summary = df.groupby('AI_Sentiment_Tag')['Rating'].mean().reset_index()
    print(summary)
    
    df.to_csv("sentiment_analysis_output.csv", index=False)
    print("\nAnalysis complete. Output saved to 'sentiment_analysis_output.csv'.")

if __name__ == "__main__":
    analyze_feedback()
