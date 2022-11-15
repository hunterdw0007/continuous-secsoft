from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model
import pandas as pd
import numpy as np 
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import logging_example

# logging initialized
logger = logging_example.getSQALogger()

def readData():
    iris = datasets.load_iris()
    # logging iris - preventing poisoning
    logger.debug('{}*{}*{}*{}'.format('workshop9.py', 'readData', type(iris.data), type(iris.target)))  
    print(type(iris.data), type(iris.target))
    X = iris.data
    Y = iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    # logging dataframe columns - preventing poisoning
    logger.debug('{}*{}*{}*{}'.format('workshop9.py', 'readData', 'dataframe', df.columns()))
    print(df.head())

    return df 

def makePrediction():
    iris = datasets.load_iris()
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(iris['data'], iris['target'])
    X = [
        [5.9, 1.0, 5.1, 1.8],
        [3.4, 2.0, 1.1, 4.8],
    ]
    prediction = knn.predict(X)
    # logging prediction - preventing model tricking
    logger.debug('{}*{}*{}*{}'.format('workshop9.py', 'readData', 'prediction', prediction))
    print(prediction)    

def doRegression():
    diabetes = datasets.load_diabetes()
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]
    regr = linear_model.LinearRegression()
    # logging linear regression model - preventing model tricking
    logger.debug('{}*{}*{}*{}'.format('workshop9.py', 'doRegression', 'Linear Regression', regr))
    regr.fit(diabetes_X_train, diabetes_y_train)
    diabetes_y_pred = regr.predict(diabetes_X_test)


def doDeepLearning():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()


    train_images = (train_images / 255) - 0.5
    test_images = (test_images / 255) - 0.5


    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    # logging images - preventing model tricking
    logger.debug('{}*{}*{}*{}'.format('workshop9.py', 'doDeepLearning', train_images, test_images))

    num_filters = 8
    filter_size = 3
    pool_size = 2

    model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10, activation='softmax'),
    ])

    # logging model - preventing model tricking
    logger.debug('{}*{}*{}*{}'.format('workshop9.py', 'doDeepLearning', 'model', model))

    # Compile the model.
    model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )

    # Train the model.
    model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=3,
    validation_data=(test_images, to_categorical(test_labels)),
    )

    model.save_weights('cnn.h5')

    predictions = model.predict(test_images[:5])

    # logging predictions - preventing model tricking
    logger.debug('{}*{}*{}*{}'.format('workshop9.py', 'doDeepLearning', 'predictions', predictions))

    print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

    print(test_labels[:5]) # [7, 2, 1, 0, 4]


if __name__=='__main__': 
    data_frame = readData()
    makePrediction() 
    doRegression() 
    doDeepLearning() 