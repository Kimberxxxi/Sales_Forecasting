import pandas as pd
import numpy as np
def extract_features(features, data):
    """
    This module deals with feature engineering.
    The parameter 'features' is a list that saves the feature names; 'data' represents the dataframe from which features will be extracted.
    """
    # Check if the input dataframe is empty or None
    if data is None or data.empty:
        raise ValueError("Input dataframe is empty or None.")
    
    # Ensure that required columns exist in the dataframe
    required_columns = ['Year', 'Promo2SinceYear', 'WeekOfYear', 'Promo2SinceWeek', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'PromoInterval', 'Set', 'Sales', 'Customers', 'Open', 'SchoolHoliday', 'Store', 'DayOfWeek']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate 'PromoOpenInMonth' and handle potential issues
    try:
        features.append('PromoOpenInMonth')
        data['PromoOpenInMonth'] = 12 * (data['Year'] - data['Promo2SinceYear']) + (data['WeekOfYear'] - data['Promo2SinceWeek']) / 4.0
        data['PromoOpenInMonth'] = data['PromoOpenInMonth'].apply(lambda x: x if x > 0 else 0)
        data.loc[data.Promo2SinceYear == 0, 'PromoOpenInMonth'] = 0
    except Exception as e:
        raise Exception(f"Error calculating 'PromoOpenInMonth': {e}")

    # Calculate 'CompetitionOpenInMonth' and handle potential issues
    try:
        features.append('CompetitionOpenInMonth')
        data['CompetitionOpenInMonth'] = 12 * (data['Year'] - data['CompetitionOpenSinceYear']) + (data['Month'] - data['CompetitionOpenSinceMonth'])
    except Exception as e:
        raise Exception(f"Error calculating 'CompetitionOpenInMonth': {e}")

    # Calculate 'IsPromoMonth' and handle potential issues
    try:
        features.append('IsPromoMonth')
        month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
        data['monthStr'] = data['Month'].map(month2str)
        data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
        data['IsPromoMonth'] = 0
        for interval in data['PromoInterval'].unique():
            if interval != '':
                for month in interval.split(','):
                    data.loc[(data['monthStr'] == month) & (data['PromoInterval'] == interval), 'IsPromoMonth'] = 1
    except Exception as e:
        raise Exception(f"Error calculating 'IsPromoMonth': {e}")

    # Merge average sales, customers, and other features into the data
    try:
        features.extend(['AvgSales', 'AvgCustomers', 'AvgSalesPerCustomer', 'medianCustomers'])

        # 1. Get total sales, customers, and open days per store
        train_df = data[data['Set'] == 1]
        totalSalesPerStore = train_df.groupby(['Store'])['Sales'].sum()
        totalCustomersPerStore = train_df.groupby(['Store'])['Customers'].sum()
        totalOpenStores = train_df.groupby(['Store'])['Open'].count()
        medianCustomers = train_df.groupby(['Store'])['Customers'].median()

        # 2. Compute averages
        AvgSales = totalSalesPerStore / totalOpenStores
        AvgCustomers = totalCustomersPerStore / totalOpenStores
        AvgSalesPerCustomer = AvgSales / AvgCustomers

        # 3. Merge the averages into data
        data = pd.merge(data, AvgSales.reset_index(name='AvgSales'), how='left', on=['Store'])
        data = pd.merge(data, AvgCustomers.reset_index(name='AvgCustomers'), how='left', on=['Store'])
        data = pd.merge(data, AvgSalesPerCustomer.reset_index(name='AvgSalesPerCustomer'), how='left', on=['Store'])
        data = pd.merge(data, medianCustomers.reset_index(name='medianCustomers'), how='left', on=['Store'])
    except Exception as e:
        raise Exception(f"Error during feature engineering for 'AvgSales', 'AvgCustomers', etc.: {e}")

    # Return the updated dataframe with the extracted features
    return data

