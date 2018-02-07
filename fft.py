"""
program implements simple 2D fast fourier transform with tensorflow
Gabriel Meyer-Lee
"""
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import tensorflow as tf
import scipy.io as sio
from mpl_toolkits.axes_grid1 import ImageGrid

def fftshift(image):
    n = tf.shape(image)
    tsplit, bsplit = tf.split(image, [tf.to_int32(tf.ceil(n[0]/2)), tf.to_int32(tf.floor(n[0]/2))], 0)
    image = tf.concat([bsplit, tsplit], 0)
    lsplit, rsplit = tf.split(image, [tf.to_int32(tf.ceil(n[1]/2)), tf.to_int32(tf.floor(n[1]/2))], 1)
    image = tf.concat([rsplit, lsplit], 1)
    return image

def scale(image):
    min = tf.reduce_min(image)
    max = tf.reduce_max(image)
    image = (image-min)/(max-min)
    return image


def fft(k_data):
    #img = tf.spectral.ifft2d(k_data)
    img = np.fft.ifft2(k_data, axes=(0,1))
    img = tf.convert_to_tensor(img, dtype=tf.complex64)
    img = tf.reduce_sum(img, 2)
    k_map = tf.abs(tf.spectral.fft2d(img))
    return fftshift(tf.abs(img)), k_map

def par_fft(k_data):
    #imgs = tf.spectral.ifft2d(tf.spectral.fft(k_data))
    imgs = np.fft.ifft2(k_data, axes=(0,1))
    imgs = tf.convert_to_tensor(imgs, dtype=tf.complex64)
    mag = fftshift(tf.abs(imgs))
    mag = scale(mag)
    phase = fftshift(tf.angle(imgs))
    return mag, phase

if __name__=='__main__':
    mat_contents = sio.loadmat('phantom_data.mat')
    k_data = tf.convert_to_tensor(mat_contents['kData'], dtype=tf.complex64)
    with tf.Session() as sess:
        image, k_map = fft(mat_contents['kData'])
        k_mag = (tf.abs(k_data))
        mag, phase = par_fft(mat_contents['kData'])
        par_imag = sess.run([mag, phase])
        k_mag = sess.run(k_mag)
        image = sess.run([image, k_map])

    fig1 = plt.figure(1)
    plt.title('Standard reconstructed image and k-space image')
    plt.axis('off')
    grid = ImageGrid(fig1, 111, nrows_ncols=(1, 2), axes_pad=0.5,)
    norm1 = [clr.Normalize(), clr.Normalize()]
    for i in range(2):
        grid[i].imshow(image[i],cmap=plt.cm.binary_r,norm=norm1[i])
        grid[i].axis('off')
        grid[i].set_xticks([])
        grid[i].set_yticks([])
    fig2 = plt.figure(2)
    plt.title('Parallel k-space images')
    plt.axis('off')
    grid2 = ImageGrid(fig2, 111, nrows_ncols=(4, 4), axes_pad=0,)
    norm2 = clr.Normalize()
    for i in range(16):
        grid2[i].imshow(k_mag[:,:,i],cmap=plt.cm.binary_r, norm=norm2)
        grid2[i].axis('off')
        grid2[i].set_xticks([])
        grid2[i].set_yticks([])
    fig3 = plt.figure(3)
    plt.title('Parallel reconstructed images')
    plt.axis('off')
    norm3 = clr.Normalize()
    grid3 = ImageGrid(fig3, 111, nrows_ncols=(4, 4), axes_pad=0,)
    for i in range(16):
        grid3[i].imshow(par_imag[0][:,:,i],cmap=plt.cm.binary_r, norm=norm3)
        grid3[i].axis('off')
        grid3[i].set_xticks([])
        grid3[i].set_yticks([])
    fig4 = plt.figure(4)
    plt.title('Parallel phase images')
    plt.axis('off')
    grid4 = ImageGrid(fig4, 111, nrows_ncols=(4, 4), axes_pad=0,)
    norm4 = clr.Normalize()
    for i in range(16):
        grid4[i].imshow(par_imag[1][:,:,i],cmap=plt.cm.binary_r,norm=norm4)
        grid4[i].axis('off')
        grid4[i].set_xticks([])
        grid4[i].set_yticks([])
    plt.show()
