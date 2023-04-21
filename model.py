import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, Conv2DTranspose, Input, Conv2D, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

latent_dim = 100
n_epochs = 100
n_batch = 64
input_images = Input(shape=(100), name='input_images')
height = 28
width = 28
channels = 3
img_shape = (height, width, channels)

#create teslorflow session
session = tf.compat.v1.Session()

#generator will take in a random noise vector and generate an image
def build_generator(latent_dim):
  model = Sequential()
  model.add(Dense(7*7*256, use_bias=False, input_shape=(latent_dim,)))
  model.add(BatchNormalization())
  model.add(LeakyReLU())

  model.add(Reshape((7, 7, 256)))
  assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

  model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
  assert model.output_shape == (None, 7, 7, 128)
  model.add(BatchNormalization())
  model.add(LeakyReLU())

  model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 14, 14, 64)
  model.add(BatchNormalization())
  model.add(LeakyReLU())

  model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
  assert model.output_shape == (None, 28, 28, 1)

  # Reshape input to be (None, 100)
  input_tensor = Input(shape=(latent_dim,))
  output_tensor = model(input_tensor)

  return Model(inputs=input_tensor, outputs=output_tensor)

#discriminator will take in an image and output a probability of whether it is real or fake
def build_discriminator(img_shape):
  model = Sequential()
  input_tensor = Input(shape=img_shape)
  x = Flatten()(input_tensor)
  x = Dense(128, activation='relu')(x)
  
  model.add(Conv2D(32, kernel_size=3, strides=(2, 2), padding='same', input_shape=img_shape))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, kernel_size=3, strides=(2, 2), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))

  model.add(Conv2D(128, kernel_size=3, strides=(2, 2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))

  model.add(Conv2D(256, kernel_size=3, strides=(1, 1), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))
  return model

#build the GAN
def build_gan(generator, discriminator):
  #compile discriminator
  discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

  #compile generator
  generator.compile(loss='binary_crossentropy', optimizer=Adam())

  #compile model
  gan = Sequential()
  gan.add(generator)
  gan.add(discriminator)
  gan.compile(loss='binary_crossentropy', optimizer=Adam())
  return gan

# add model output tensor
def build_output_tensor(discriminator):
  return discriminator.output

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
          axs[i,j].imshow(images[count, :, :, 0], cmap='gist_rainbow_r')
          axs[i,j].axis('off')
          count += 1
  # plt.savefig(f'{i}', f'gan_generated_image{i}.png')
  plt.show(block=True)

#train the GAN

def train(gan, discriminator, generator, latent_dim, n_epochs=100, n_batch=64):
  half_batch = int(n_batch / 2)
  for i in range(n_epochs):
    #generate real and fake samples
    x_real, y_real = generate_real_samples(half_batch)
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)
    # train discriminator
    discriminator.train_on_batch(x_real, y_real)
    discriminator.train_on_batch(x_fake, y_fake)
    # generate noise vectors and labels for gan training
    x_gan = generate_latent_points(latent_dim, n_batch)
    y_gan = np.ones((n_batch, 1))

    # train gan on noise vectors and "real" labels
    gan.train_on_batch(x_gan, y_gan)

    if (i+1) % 10 == 0:
      print(f"Epoch {i+1}/{n_epochs}")
      x_fake, _ = generate_fake_samples(generator, latent_dim, 25)
      plot_images(x_fake)

#datasets
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

#normalize pixel values
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=3)
x_train = np.repeat(x_train, 3, axis=3)

#build the generator
generator = build_generator(latent_dim)

#build the discriminator
discriminator = build_discriminator(img_shape=(28,28,1))

input_latent = Reshape((latent_dim,))(input_images)
#build the gan


graph = tf.Graph()

generated_images_tensor = generator(input_latent)
model_output_tensor = discriminator(generated_images_tensor)

gan_input_images = Input(shape=(latent_dim), name='gan_input_images')
gan_output = discriminator(generator(gan_input_images))

gan = build_gan(generator, discriminator)
## or do i want this
gan2 = Model(gan_input_images, gan_output)

gan3 = Model(input_images, model_output_tensor)



# with graph.as_default():
#   placeholder_tensor = tf.compat.v1.placeholder(tf.float32, shape=(None, 100), name='Placeholder/_1')
#   generated_images_tensor = generator(placeholder_tensor)
#   model_output_tensor = discriminator(generated_images_tensor)

# with tf.compat.v1.Session(graph=graph) as session:
#   session.run(tf.compat.v1.global_variables_initializer())
#   output = session.run(model_output_tensor, feed_dict={placeholder_tensor: x_train})

generator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
gan.compile(loss='binary_crossentropy', optimizer='adam')
gan2.compile(loss='binary_crossentropy', optimizer='adam')
gan3.compile(loss='binary_crossentropy', optimizer='adam')

#train the model
#trains gan for 'n_epochs' epochs, using batches of size n_batch
# during training, gan will generate fake images to try to fool discriminator
# discriminator will be trained on both real and fake images to learn to distinguish between them
train(gan, discriminator, generator, latent_dim, n_epochs=n_epochs, n_batch=n_batch)
train(gan2, discriminator, generator, latent_dim, n_epochs=n_epochs, n_batch=n_batch)
train(gan3, discriminator, generator, latent_dim, n_epochs=n_epochs, n_batch=n_batch)

#generate 25 random images
latent_points = generate_latent_points(latent_dim, 25)
generated_images = generator.predict(latent_points)
plot_images(generated_images)

