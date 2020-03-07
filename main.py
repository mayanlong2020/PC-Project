import stock

print(dir(stock))
data = stock.loaddata()
print(data)

#X_train, y_train, X_test, y_test = stock.preprocess(data)

X_train, y_train, X_test, y_test = stock.preprocess(data)
model = Sequential()
stock.training(X_train, y_train)

#stock.valiating(X_test, y_test)

#stock.predicting(X, y_actual)
