import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
data =pd.read_csv("HR_comma_sep.csv")
# print(data)
#step 1 missing data
# print(data.isnull().values.any())

# step 2  check data types
# print(data.dtypes)

# step 3 check unique types
# print(data.department.unique())

# convert string to numeric values
clean_up_values={
      "salary":
            {"low":1,
             "medium":2,
             "high":3}
            }

data.replace(clean_up_values,inplace=True)

# print(data.salary.unique())

# step 4 get dummies for the department
dummies= pd.get_dummies(data.department)
# print(dummies)

# step 5 merge dummies with the original data
merged= pd.concat([data,dummies],axis="columns")
# print(merged)

#step 6 drop unneccesary column
final_data= merged.drop(["department"],axis="columns")
# print(final_data.columns)
# step 7 show
# plt.scatter(x=final_data.satisfaction_level,y=final_data.left)
# plt.show()

x=final_data.drop("left",axis="columns")
y = final_data.left

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model =LogisticRegression()
model.fit(x_train,y_train)

acuurecy = model.score(x_test,y_test)

print(acuurecy)