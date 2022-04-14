import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import os
import sys
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier

# Google Colab stuff
# from google.colab import drive
# drive.mount('/content/drive')


def train_test_split(x, y, train_proportion):
    t = np.size(y)*train_proportion
    t = int(t)
    x_train = x[:t]
    x_test = x[t:]
    y_train = y[:t]
    y_test = y[t:]
    return x_train, x_test, y_train, y_test


def get_data(datafile):
    dataframe = pd.read_csv(datafile)
    data = list(dataframe.values)
    labels, images = [], []
    for line in data:
        labels.append(line[0])
        images.append(line[1:])
    labels = np.array(labels)
    images = np.array(images).astype('float32')
    images /= 255
    return images, labels


def visualize_weights(trained_model, num_to_display=20, save=True, hot=True):
    layer1 = trained_model.layers[0]
    weights = layer1.get_weights()[0]

    colors = 'hot' if hot else 'binary'
    try:
        os.mkdir('weight_visualizations')
    except FileExistsError:
        pass
    for i in range(num_to_display):
        wi = weights[:, i].reshape(28, 28)
        plt.imshow(wi, cmap=colors, interpolation='nearest')
        if save:
            plt.savefig('./weight_visualizations/unit' +
                        str(i) + '_weights.png')
        else:
            plt.show()


def output_predictions(predictions, model_type):
    if model_type == 'CNN':
        with open('CNNpredictions.txt', 'w+') as f:
            for pred in predictions:
                f.write(str(pred) + '\n')
    if model_type == 'MLP':
        with open('MLPpredictions.txt', 'w+') as f:
            for pred in predictions:
                f.write(str(pred) + '\n')


def plot_history(history):
    train_loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']

    train_acc_history = history.history['accuracy']
    val_acc_history = history.history['val_accuracy']

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(val_acc_history, color="red",
             marker='o', label="validation accuracy")
    ax1.plot(train_acc_history, color="blue",
             marker='o', label="training accuracy")
    ax1.set_title("Training and Validation accuracy vs Epoch")
    ax1.legend(loc="upper left")
    ax2.plot(val_loss_history, color="red",
             marker='o', label="validation loss")
    ax2.plot(train_loss_history, color="blue",
             marker='o', label="training loss")
    ax2.set_title("Training and validation loss vs epoch")
    ax2.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def create_mlp(args=None):
    # Define model architecture
    model = Sequential()
    model.add(Dense(units=784, activation="relu", input_dim=28*28))
    model.add(Dense(units=512, activation="relu",))
    model.add(Dense(units=128, activation="relu",))
    model.add(Dense(units=64, activation="relu",))
    model.add(Dense(units=32, activation="relu",))
    model.add(Dense(units=10, activation="sigmoid",))

    # Optimizer
    if args['opt'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr=args['learning_rate'])
    elif args['opt'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=args['learning_rate'])

    # Compile
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=optimizer, metrics=['accuracy'])

    return model


def train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=None):
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    model = create_mlp(args)
    history = model.fit(
        x_train, y_train, batch_size=args['batch_size'], epochs=args['epoch'], validation_split=0.01)
    return model, history


def create_cnn(args=None):

    # 28x28 images with 1 color channel
    input_shape = (28, 28, 1)

    # Define model architecture

    model = Sequential()
    model.add(Conv2D(filters=3, activation="relu", kernel_size=(
        6, 6), strides=1, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=1))
    # model.add(Conv2D(filters=3, activation="relu", kernel_size=(3,3), strides=1, input_shape=input_shape))
    # model.add(MaxPooling2D(pool_size=(2,2), strides=1))

    model.add(Flatten())

    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=10, activation="softmax"))

    # Optimizer
    if args['opt'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr=args['learning_rate'])
    elif args['opt'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=args['learning_rate'])

    # Compile
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=optimizer, metrics=['accuracy'])

    return model


def train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=None):
    x_train = x_train.reshape(-1, 28, 28, 1)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    model = create_cnn(args)
    history = model.fit(
        x_train, y_train, batch_size=args['batch_size'], epochs=args['epoch'], validation_split=0.01)
    return model, history


def train_and_select_model(train_csv, model_type, grading_mode):

    x_train, y_train = get_data(train_csv)

    args = {
        'batch_size': 128,
        'validation_split': 0.01,
        'epoch': 15
    }

    best_valid_acc = 0
    best_hyper_set = {}
    other_hyper_set = {'opt', 'learning_rate', 'other_hyper'}

    if not grading_mode:
        for learning_rate in [0.05, 0.01, 0.005]:
            # for learning_rate in [0.01]:
            for opt in ['adam', 'sgd']:
                for other_hyper in other_hyper_set:  # search over other hyperparameters
                    args['opt'] = opt
                    args['learning_rate'] = learning_rate
                    args['other_hyper'] = other_hyper

                    if model_type == 'MLP':
                        model, history = train_mlp(
                            x_train, y_train, x_vali=None, y_vali=None, args=args)
                    else:
                        model, history = train_cnn(
                            x_train, y_train, x_vali=None, y_vali=None, args=args)

                    validation_accuracy = history.history['val_accuracy']

                    max_valid_acc = max(validation_accuracy)
                    if max_valid_acc > best_valid_acc:
                        best_model = model
                        best_valid_acc = max_valid_acc
                        best_hyper_set['learning_rate'] = learning_rate
                        best_hyper_set['opt'] = opt
                        best_history = history
    else:
        if model_type == 'MLP':
            args['opt'] = "adam"
            args['learning_rate'] = 0.001

            args['hidden_dim'] = 28
            args['hidden_layer'] = 2
            args['activation'] = "relu"

            best_model, best_history = train_mlp(
                x_train, y_train, x_vali=None, y_vali=None, args=args)

        if model_type == 'CNN':
            args['opt'] = "adam"
            args['learning_rate'] = 0.001
            best_model, best_history = train_cnn(
                x_train, y_train, x_vali=None, y_vali=None, args=args)

    return best_model, best_history


if __name__ == '__main__':
    grading_mode = True
    if grading_mode:

        if (len(sys.argv) != 3):
            print("Usage:\n\tpython3 fashion.py train_file test_file")
            exit()
        train_file, test_file = sys.argv[1], sys.argv[2]

        # train best model
        # best_mlp_model, mlp_history = train_and_select_model(
        #     train_file, model_type='MLP', grading_mode=True)

        # print("PCA+LOGREG")
        # xtrain, ytrain = get_data(train_file)
        # xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain,0.5)
        # errors = {}
        # components = []
        # for i in (2,5,10,50,100) :
        #     components.append(i)
        #     pca = PCA(n_components=i)
        #     logr = LogisticRegression (max_iter=2000, solver = 'saga')
        #     scaler = StandardScaler()
        #     #pipe = Pipeline ([('scaler',scaler), ('logistic', logr)])
        #     pipe = Pipeline ([('scaler',scaler), ('pca', pca), ('logistic', logr)])
        #     pipe.fit(xtrain, ytrain)
        #     pred = pipe.predict (xvalid)
        #     errors[i] = 1 - accuracy_score(yvalid, pred)
        #     print(accuracy_score(yvalid, pred))
        # plt.plot(components, list(errors.values()), color = "blue", marker = 'o')
        # plt. title( 'PCA + Logistic Regression')
        # plt.xlabel ('Principal Components')
        # plt.ylabel ('Error')
        # plt. show()
        # print("PCA+LOGREG")

        # print("LOGREG ONLY")
        # x_train, y_train = get_data(train_file)
        # X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train,0.5)
        # pca = PCA(n_components=5)
        # clf = LogisticRegression(max_iter=6000, solver = 'saga')
        # scaler = StandardScaler()
        # pipe = Pipeline([('scaler',scaler),('logistic', clf)])
        # pipe.fit(X_train, y_train)
        # predictions = pipe.predict(X_valid)
        # print(1-accuracy_score(predictions,y_train))
        # print("LOGREG ONLY")

        x_test, _ = get_data(test_file)
        # generate predictions for test_file
        # mlp_predictions = best_mlp_model.predict(x_test)
        # print(best_mlp_model.summary())
        # mlpargmax = []
        # for line in mlp_predictions:
        #     mlpargmax.append(np.argmax(line))
        # output_predictions(mlpargmax, model_type='MLP')

        # visualize_weights(best_mlp_model)
        # plot_history(mlp_history)

        x_test = x_test.reshape(-1, 28, 28, 1)
        best_cnn_model, cnn_history = train_and_select_model(
            train_file, model_type='CNN', grading_mode=True)
        cnn_predictions = best_cnn_model.predict(x_test)
        cnnargmax = []
        for line in cnn_predictions:
            cnnargmax.append(np.argmax(line))
        output_predictions(cnnargmax, model_type='CNN')
        # print(best_cnn_model.summary())
        # plot_history(cnn_history)

    else:

        train_file = '/content/drive/My Drive/ML/fashion_train.csv'
        test_file = '/content/drive/My Drive/ML/fashion_test_labeled.csv'
        # MLP
        mlp_model, mlp_history = train_and_select_model(
            train_file, model_type='MLP', grading_mode=False)
        plot_history(mlp_history)
        visualize_weights(mlp_model)

        # CNN
        cnn_model, cnn_history = train_and_select_model(
            train_file, model_type='CNN', grading_mode=False)
        plot_history(cnn_history)
