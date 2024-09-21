import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from xgboost import XGBRegressor
import category_encoders as ce

# Загрузка данных
train_data = pd.read_csv('train.csv', engine="pyarrow")
validation_data = pd.read_csv('test_merged.csv', engine="pyarrow")

# Логарифмирование целевой переменной
train_data['price'] = np.log1p(train_data['price']).astype(np.float64)
validation_data['price'] = np.log1p(validation_data['price']).astype(np.float64)

# Создание новых признаков
current_year = 2024  # Замените на текущий год или извлеките из данных
train_data['car_age'] = current_year - train_data['model_year']
train_data['mileage_per_year'] = train_data['milage'] / (train_data['car_age'] + 1)
validation_data['car_age'] = current_year - validation_data['model_year']
validation_data['mileage_per_year'] = validation_data['milage'] / (validation_data['car_age'] + 1)


# Обновление списков признаков
numerical_columns = ['model_year', 'milage', 'car_age', 'mileage_per_year']
categorical_columns = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col', 'accident',
                       'clean_title']

# Разделение данных на X и y
X = train_data.drop(columns=['price', 'id'])
y = train_data['price']
X_valid = validation_data.drop(columns=['price', 'id'])
y_valid = validation_data['price']
# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.13, random_state=42)

# Определение признаков с высокой кардинальностью
high_cardinality_cols = ['model', "brand"]  # Предположим, что 'model' имеет высокую кардинальностью
low_cardinality_cols = list(set(categorical_columns) - set(high_cardinality_cols))

# Препроцессоры
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

low_cardinality_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

high_cardinality_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('target_encoder', ce.TargetEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('low_cat', low_cardinality_transformer, low_cardinality_cols),
        ('high_cat', high_cardinality_transformer, high_cardinality_cols)
    ])

# Создание пайплайна с XGBoost с фиксированными гиперпараметрами
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        colsample_bytree=0.6092202186084168,
        learning_rate=0.09330198957407325,
        alpha= 0.0016104367941295323,
        max_depth=12,
        n_estimators=430,
        subsample=0.662172510502637
    ))
])

# Обучение модели
pipeline.fit(X_train, y_train)

# Прогнозы на тестовом наборе
y_valid_pred = pipeline.predict(X_valid)

# Обратное преобразование логарифмической функции
y_valid_pred_exp = np.expm1(y_valid_pred)

# Создание DataFrame с id и предсказанными значениями цены
output_df = pd.DataFrame({
    'id': validation_data['id'],
    'predicted_price': y_valid_pred_exp
})

# Сохранение результата в CSV файл
output_df.to_csv('validation_predictions.csv', index=False)

print('Файл с предсказаниями сохранен: validation_predictions.csv')

