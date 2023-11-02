# MNIST [RU]
#
# Описание проекта
Этот проект представляет собой пример реализации нейронных сетей для классификации изображений с использованием библиотеки PyTorch. Проект включает две реализации моделей: полносвязную нейронную сеть и сверточную нейронную сеть. Каждая модель обучается на наборе данных MNIST для распознавания рукописных цифр.

# Полносвязная нейронная сеть
В этой реализации используется простая полносвязная нейронная сеть с несколькими слоями. Используются различные функции активации, такие как ELU, ReLU и LeakyReLU, чтобы сравнить их влияние на обучение.

# Сверточная нейронная сеть (LeNet)
Сверточная нейронная сеть LeNet - это более сложная модель, специально разработанная для анализа изображений. Она состоит из сверточных и полносвязанных слоев, и она также обучается на наборе данных MNIST.

# Установка
Для запуска проекта вам понадобятся следующие библиотеки:

PyTorch
torchvision
torchsummary
matplotlib (для построения графиков)
seaborn (для улучшения визуализации данных)
Убедитесь, что у вас установлены все необходимые библиотеки, прежде чем запустить код.

# Запуск проекта
Загрузите код из репозитория.
Откройте Jupyter Notebook или другое средство разработки для Python.
Запустите код из файла, чтобы обучить модели и оценить их точность.
# Результаты
После завершения обучения моделей вы увидите графики точности на данных для обучения и валидации. Это позволит вам сравнить производительность разных функций активации и моделей.
#
#
#
#
#
# MNIST [ENG]
# Project Description
This project is an example implementation of neural networks for image classification using the PyTorch library. The project includes two model implementations: a fully connected neural network and a convolutional neural network. Each model is trained on the MNIST dataset for handwritten digit recognition.

# Fully Connected Neural Network
In this implementation, a simple fully connected neural network with multiple layers is used. Various activation functions such as ELU, ReLU, and LeakyReLU are employed to compare their impact on training.

# Convolutional Neural Network (LeNet)
The LeNet convolutional neural network is a more complex model designed specifically for image analysis. It consists of convolutional and fully connected layers, and it is also trained on the MNIST dataset.

# Installation
To run the project, you will need the following libraries:

PyTorch
torchvision
torchsummary
matplotlib (for plotting)
seaborn (for improved data visualization)
Make sure you have all the necessary libraries installed before running the code.

# Running the Project
Download the code from the repository.
Open a Jupyter Notebook or another Python development environment.
Run the code from the file to train the models and evaluate their accuracy.
# Results
After the models have finished training, you will see accuracy plots for both the training and validation data. This allows you to compare the performance of different activation functions and models.
