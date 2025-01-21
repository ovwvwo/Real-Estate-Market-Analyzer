import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class RealEstateAnalyzer:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_importance = None
        
    def generate_sample_data(self, n_samples=1000):
        """Генерация тестового набора данных о недвижимости"""
        np.random.seed(42)
        
        data = {
            'area': np.random.normal(60, 20, n_samples),
            'rooms': np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.4, 0.4, 0.1]),
            'floor': np.random.randint(1, 25, n_samples),
            'total_floors': np.random.randint(5, 25, n_samples),
            'year_built': np.random.randint(1960, 2024, n_samples),
            'distance_to_center': np.random.normal(7, 3, n_samples),
            'district': np.random.choice(['Центральный', 'Северный', 'Южный', 'Западный', 'Восточный'], n_samples),
            'condition': np.random.choice(['Новый', 'Хороший', 'Требует ремонта'], n_samples),
            'parking': np.random.choice(['Есть', 'Нет'], n_samples)
        }
        
        base_price = 150000
        price = (
            base_price * data['area'] * 
            (1 + 0.1 * (data['rooms'] - 1)) * 
            (1 - 0.01 * (2024 - data['year_built'])) * 
            (1 - 0.05 * (data['distance_to_center'] / 5)) * 
            np.random.normal(1, 0.1, n_samples)
        )
        
        data['price'] = price
        data['total_floors'] = np.maximum(data['total_floors'], data['floor'])
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        numeric_features = ['area', 'rooms', 'floor', 'total_floors', 'year_built', 
                            'distance_to_center']
        categorical_features = ['district', 'condition', 'parking']
        
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse=False)
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return self.preprocessor
    
    def train_model(self, df, target='price'):
        X = df.drop(target, axis=1)
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = Pipeline([
            ('preprocessor', self.preprocess_data(df)),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return metrics, (X_test, y_test, y_pred)
    
    def analyze_feature_importance(self, df, target='price'):
        feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.named_steps['regressor'].feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    
    def create_visualizations(self, df):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        sns.boxplot(data=df, x='district', y='price', ax=axes[0, 0])
        axes[0, 0].set_title('Распределение цен по районам')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        sns.scatterplot(data=df, x='area', y='price', hue='rooms', ax=axes[0, 1])
        axes[0, 1].set_title('Зависимость цены от площади')
        
        df.groupby('year_built')['price'].mean().plot(ax=axes[1, 0])
        axes[1, 0].set_title('Средняя цена по году постройки')
        
        sns.histplot(data=df, x='price', bins=50, ax=axes[1, 1])
        axes[1, 1].set_title('Распределение цен')
        
        plt.tight_layout()
        return fig
    
    def predict_price(self, features):
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала выполните train_model()")
        
        input_df = pd.DataFrame([features])
        prediction = self.model.predict(input_df)[0]
        return prediction

if __name__ == "__main__":
    analyzer = RealEstateAnalyzer()
    data = analyzer.generate_sample_data()
    metrics, _ = analyzer.train_model(data)
    print(f"R² score: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:,.0f}, MAPE: {metrics['mape']:.1f}%")
    feature_importance = analyzer.analyze_feature_importance(data)
    print(feature_importance)
    analyzer.create_visualizations(data)
    plt.show()
