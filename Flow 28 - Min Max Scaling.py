import pandas as pd


# Scaling training frame
train = pd.read_csv('fe_train_imputed.csv')
scaling_columns = list(train.columns)
scaling_columns.remove('ndvi_ne')
scaling_columns.remove('ndvi_nw')
scaling_columns.remove('ndvi_se')
scaling_columns.remove('ndvi_sw')
scaling_columns.remove('city')
scaling_columns.remove('year')
scaling_columns.remove('week_start_date')
scaling_columns.remove('total_cases')

print("Scaling started...")
for column in scaling_columns:
    print("Scaling : " + column)
    print("----------------------------------")
    minimum = min(list(train[column]))
    maximum = max(list(train[column]))
    diff = maximum - minimum
    print("Max     : " + str(maximum))
    print("Min     : " + str(minimum))
    print("Diff    : " + str(diff))
    train[column] = (train[column] - minimum) / diff
    print("----------------------------------")

print("Saving : fe_train_imputed_minmax.csv")
train.to_csv('fe_train_impute_minmax.csv', index=False)

# Scaling testing frame
test = pd.read_csv('fe_test_imputed.csv')
scaling_columns = list(test.columns)
scaling_columns.remove('ndvi_ne')
scaling_columns.remove('ndvi_nw')
scaling_columns.remove('ndvi_se')
scaling_columns.remove('ndvi_sw')
scaling_columns.remove('city')
scaling_columns.remove('year')
scaling_columns.remove('week_start_date')

print("Scaling started...")
for column in scaling_columns:
    print("Scaling : " + column)
    print("----------------------------------")
    minimum = min(list(test[column]))
    maximum = max(list(test[column]))
    diff = maximum - minimum
    print("Max     : " + str(maximum))
    print("Min     : " + str(minimum))
    print("Diff    : " + str(diff))
    test[column] = (test[column] - minimum) / diff
    print("----------------------------------")

print("Saving : fe_test_imputed_minmax.csv")
test.to_csv('fe_test_impute_minmax.csv', index=False)


