from src.components.data_ingestion import DataIngestion
from src.components.preprocessing import DataPreprocessing
from src.components.prediction import SentimentPredictions
import os
import seaborn as sns
import matplotlib.pyplot as plt 

product_name='iphone13'
obj=DataIngestion(product_name)
obj.initiate_data_ingestion() 
obj1=DataPreprocessing('static\{}.csv'.format(product_name))
obj1.preprocessing()
obj2=SentimentPredictions('static\processed_1.csv')
X,Y,Z = obj2.predictions()
print(X,Y,Z)
labels = ['Positive', 'Neutral', 'Negative']
sizes = [X, Y, Z]  # Example sizes for the pie chart

# Create a pie plot using Seaborn
plt.figure(figsize=(12, 12))
sns.set_style("whitegrid")
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal') 
os.makedirs('static',exist_ok=True)
image_path = os.path.join('static','pie_chart.png')
print(image_path)
plt.savefig(image_path, bbox_inches='tight')
