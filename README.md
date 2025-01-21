# Real-Estate-Market-Analyzer
Comprehensive Python class for real estate market analysis, featuring data generation, price prediction, and visualization capabilities. The analyzer uses Random Forest to predict property prices based on multiple factors and provides detailed insights through statistical analysis and visualizations.

:see_no_evil: "This project is a perfect place to explore the concept of 'works on my computer'." :hear_no_evil:
## Features

### Data Generation and Processing
- Generate synthetic real estate data with realistic distributions
- Handle both numerical and categorical features
- Automatic data preprocessing including:
  - Standard scaling for numerical features
  - One-hot encoding for categorical features
  - Pipeline-based data transformation

### Price Prediction
- Random Forest Regression model
- Model evaluation metrics:
  - R-squared (R²)
  - Root Mean Square Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
- Feature importance analysis

### Visualization
- Price distribution by district (box plots)
- Price vs. area correlation with room count overlay
- Average price trends by construction year
- Overall price distribution analysis

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

### Basic Example

```python
from real_estate_analyzer import RealEstateAnalyzer

# Initialize analyzer
analyzer = RealEstateAnalyzer()

# Generate sample data
data = analyzer.generate_sample_data(n_samples=1000)

# Train model and get metrics
metrics, test_data = analyzer.train_model(data)

# Create visualizations
fig = analyzer.create_visualizations(data, test_data)

# Make predictions
sample_features = {
    'area': 65,
    'rooms': 2,
    'floor': 5,
    'total_floors': 12,
    'year_built': 2010,
    'distance_to_center': 5,
    'district': 'Центральный',
    'condition': 'Хороший',
    'parking': 'Есть'
}
predicted_price = analyzer.predict_price(sample_features)
```

## Class Methods

### `__init__()`
Initializes the RealEstateAnalyzer with empty model and preprocessor.

### `generate_sample_data(n_samples=1000)`
Generates synthetic real estate data with realistic distributions.

Parameters:
- `n_samples`: Number of samples to generate (default: 1000)

Returns:
- pandas DataFrame with generated data

### `preprocess_data(df)`
Creates a preprocessing pipeline for data transformation.

Parameters:
- `df`: Input DataFrame

Returns:
- ColumnTransformer preprocessor

### `train_model(df, target='price')`
Trains the Random Forest model for price prediction.

Parameters:
- `df`: Training data DataFrame
- `target`: Target variable name (default: 'price')

Returns:
- Tuple of (metrics dictionary, test data tuple)

### `analyze_feature_importance(df, target='price')`
Analyzes feature importance in price prediction.

Parameters:
- `df`: Input DataFrame
- `target`: Target variable name (default: 'price')

Returns:
- DataFrame with feature importance scores

### `create_visualizations(df, test_predictions=None)`
Creates a set of visualizations for data analysis.

Parameters:
- `df`: Input DataFrame
- `test_predictions`: Optional tuple of test data and predictions

Returns:
- matplotlib Figure object

### `predict_price(features)`
Predicts price for new properties.

Parameters:
- `features`: Dictionary of property features

Returns:
- Predicted price value

## Property Features

The model considers the following features:

| Feature | Type | Description |
|---------|------|-------------|
| area | float | Living space in square meters |
| rooms | int | Number of rooms (1-4) |
| floor | int | Current floor number |
| total_floors | int | Total floors in building |
| year_built | int | Construction year |
| distance_to_center | float | Distance to city center (km) |
| district | str | District name |
| condition | str | Property condition |
| parking | str | Parking availability |

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Example Output

```python
# Model metrics example
metrics = {
    'r2': 0.945,
    'rmse': 125000,
    'mape': 8.5
}

# Feature importance example
importance = {
    'area': 0.35,
    'distance_to_center': 0.25,
    'year_built': 0.15,
    'district': 0.10,
    'rooms': 0.08,
    'floor': 0.04,
    'condition': 0.02,
    'parking': 0.01
}
```

## License

This project is licensed under the MIT License.
