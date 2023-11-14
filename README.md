# cs231n_battery_proj (Archived)
Convert time series data for voltage, current and temperature for 18650 Li-ion batteries into images and run through CNN for prediction.\.  
Used Oxford Battery Degradation dataset [Birkl, C. (2017). Oxford Battery Degradation Dataset 1. University of Oxford.](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac)


### Miscellaneous
#### Between using PIL.Image vs matplotlib -> NumPy arrays
On reading in .png files created using [plt.imsave](https://matplotlib.org/stable/api/image_api.html#matplotlib.image.imread) and [Image.save](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open) (converting the Image object to np array using [matplotlib.pil_to_array](https://matplotlib.org/stable/api/image_api.html#matplotlib.image.imread) and [np.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html) both), noticed that both resulting arrays are equal, but both do not match input numpy RGB array. This is because of internal conversion of the numpy array to an Image object, since [plt.imsave calls PIL.Image.Image.save](https://matplotlib.org/stable/api/image_api.html#matplotlib.image.imsave). This can be confirmed by running the make_im_Image function and testing its output against the read file's numpy array.\
We need NumPy arrays to store RGB arrays instead of Image objects, because PyTorch models need NumPy arrays as inputs. PIL recommends using the np.asarray/np.array function for array conversions [SO post](https://stackoverflow.com/questions/384759/how-do-i-convert-a-pil-image-into-a-numpy-array), [PIL docs 1.1.6](https://web.archive.org/web/20081225061956/http://effbot.org/zone/pil-changes-116.htm). Since nearly all of the used matplotlib functions internally use PIL directly or indirectly via other modules like ImageIO, we will use the make_im_Image function and convert Image objects to np arrays to be consistent and avoid pixel value errors due to conversions.\
There also exist other resources for reading/ writing images such as:
* [OpenCV](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html) 
* [ImageIO](https://github.com/imageio/imageio)
