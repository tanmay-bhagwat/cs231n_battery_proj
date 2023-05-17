# cs231n_battery_proj
Convert time series data for voltage, current and temperature for 18650 Li-ion batteries into images and run through CNN for prediction
Used Oxford Battery Degradation dataset [Birkl, C. (2017). Oxford Battery Degradation Dataset 1. University of Oxford.](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac).

# Between using PIL.Image vs matplotlib -> NumPy arrays
We need NumPy arrays to store RGB arrays instead of Image objects, because PyTorch models need NumPy arrays as inputs. PIL recommends using the np.asarray/np.array function for array conversions [SO post](https://stackoverflow.com/questions/384759/how-do-i-convert-a-pil-image-into-a-numpy-array), [PIL docs 1.1.6](https://web.archive.org/web/20081225061956/http://effbot.org/zone/pil-changes-116.htm). Anyway, the main reason for conversion of NumPy arrays to Image objects was for data visualization and confirming all preprocessing operations were working okay.
There also exist other resources for reading/ writing images such as:
* [OpenCV](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html) 
* [ImageIO](https://github.com/imageio/imageio)
