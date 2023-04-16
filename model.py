import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Reshape, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

#generator will take in a random noise vector and generate an image
def build_generator(latent_dim):
  model = Sequential()
  model.add(Dense(128, input_dim=latent_dim, activation='relu'))
  model.add(Dense(784, activation="sigmoid"))
  model.add(Reshape((28,28, 1)))
  return model

#discriminator will take in an image and output a probability of whether it is real or fake
def build_discriminator(img_shape):
  model = Sequential()
  model.add(Flatten(input_shape=img_shape))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  return model

#build the GAN

#compile discriminator
discriminator = build_discriminator((28,28,1))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

#compile generator
generator = build_generator(100)
generator.compile(loss='binary_crossentropy', optimizer=Adam())

#compile model
gan = Sequential()
gan.add(generator)
gan.add(discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

# helper functions

def generate_latent_points(latent_dim, n_samples):
  x_input = np.random.randn(latent_dim * n_samples)
  x_input = x_input.reshape(n_samples, latent_dim)
  return x_input

def generate_real_samples(n_samples):
  (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
  x_train = x_train.astype('float32')
  x_train = x_train / 255.0
  x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
  ix = np.random.randint(0, x_train.shape[0], n_samples)
  x = x_train[ix]
  y = np.ones((n_samples, 1))
  return x, y

def generate_fake_samples(generator, latent_dim, n_samples):
  x_input = generate_latent_points(latent_dim, n_samples)
  x = generator.predict(x_input)
  y = np.zeros((n_samples, 1))
  return x, y

def plot_images(images):
  fig, axs = plt.subplots(5, 5)
  count = 0
  for i in range(5):
      for j in range(5):
          axs[i,j].imshow(images[count, :, :, 0], cmap='pink')
          axs[i,j].axis('off')
          count += 1
  plt.show()

#train the GAN

def train(gan, discriminator, generator, latent_dim, n_epochs=100, n_batch=64):
  half_batch = int(n_batch / 2)
  for i in range(n_epochs):
    x_real, y_real = generate_real_samples(half_batch)
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)
    discriminator.train_on_batch(x_real, y_real)
    discriminator.train_on_batch(x_fake, y_fake)
    x_gan = generate_latent_points(latent_dim, n_batch)
    y_gan = np.ones((n_batch, 1))
    gan.train_on_batch(x_gan, y_gan)
    if (i+1) % 10 == 0:
      print(f"Epoch {i+1}/{n_epochs}")
      x_fake, _ = generate_fake_samples(generator, latent_dim, 25)
      plot_images(generator.predict(generate_latent_points(latent_dim, 25)))

#datasets
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

#normalize pixel values
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=3)

latent_dim = 100
n_epochs = 100
n_batch = 64

#train the model
#trains gan for 'n_epochs' epochs, using batches of size n_batch
# during training, gan will generate fake images to try to fool discriminator
# discriminator will be trained on both real and fake images to learn to distinguish between them
train(gan, discriminator, generator, latent_dim, n_epochs=n_epochs, n_batch=n_batch)

#generate 25 random images
latent_points = generate_latent_points(latent_dim, 25)
generated_images = generator.predict(latent_points)
plot_images(generated_images)


