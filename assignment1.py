from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

#from IPython import get_ipython
### Config the matplotlib backend as plotting inline in IPython
#ipython_shell = get_ipython()
#ipython_shell.magic('matplotlib inline')

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

print (train_filename, test_filename)

num_classes = 10
np.random.seed(133)

def maybe_extract(filename):
    directory_name = os.path.splitext(os.path.splitext(filename)[0])[0]
    if os.path.isdir(directory_name):
        print ('Already present')
    else:
        print ('Extracting...')
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()

    data_folders = [
            os.path.join(directory_name, d) for d in os.listdir(directory_name)
            if os.path.isdir (os.path.join (directory_name, d))]

    print (data_folders)
    return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_one_letter (folder, min_num_images_per_class):
    """Load the data for a single letter label."""
    image_files = os.listdir (folder)
    dataset = np.ndarray (shape=(len(image_files), image_size, image_size), dtype=np.float32)

    print (folder)
    imageId = 0
    for image in image_files:
        image_file = os.path.join (folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth/2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape %s' % str(image_data.shape))
            dataset[imageId, : , : ] = image_data
            imageId = imageId + 1
        except IOError as e:
            print ('Could not read ', image_file, ':', e, ' It\'s OK, skipping..')
    dataset = dataset[0:imageId, :, :]
    if imageId < min_num_images_per_class:
        raise Exception('Many fewer images than expected: %d < %d' % #It's like dvm_assert_context
                                (imageId, min_num_images_per_class))
                  
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = list()
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append (set_filename)
        if (os.path.exists(set_filename) and not force):
            print ('%s already present. Skipping pickling.' %set_filename)
        else:
            print ('Pickling %s' %set_filename)
            dataset = load_one_letter (folder, min_num_images_per_class)

            try:
                with open(set_filename, "wb") as f:
                    pickle.dump (dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print ('Unable to save data to', set_filename, ':', e)

    return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

# Read back pickle file and display an image
# TODO matplotlob will NOT SHOW UNLESS matplotlib is made INLINE
#def display_random_image (pickleFile):
#    with open(pickleFile, 'rb') as f:
#        letter_set = pickle.load(f) # letter_set is a 3-dimensional array now
#        sample_imageId = np.random.randint(len(letter_set))
#        sample_image = letter_set[sample_imageId, : , : ]
#        print ('Displaying image for %s ' % pickleFile)
#        plt.figure ()
#        plt.imshow (sample_image)
#
#display_random_image(train_datasets[np.random.randint(len(train_datasets))]) # train_datasets[0] is A.pickle, train_datasets[1] is B.pickle

def make_arrays (nb_rows, img_size):
    if (nb_rows):
        dataset = np.ndarray (shape = (nb_rows, img_size, img_size), dtype = np.float32)
        labels = np.ndarray (shape = (nb_rows), dtype = np.int32)
    else:
        dataset, labels = None, None

    return dataset, labels


def merge_datasets (pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    train_dataset, train_labels = make_arrays (train_size, image_size)
    valid_dataset, valid_labels = make_arrays (valid_size, image_size)
    tsize_per_class = train_size // num_classes
    vsize_per_class = valid_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class

    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f) # Letter set is a 3D array, dim1 is imageId, dim2 and dim3 are ImageSize
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle (letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)

_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

# Finally, let's save the data for later reuse:
pickle_file = 'notMNIST.pickle'

try:
    save = {
            'train_dataset' : train_dataset,
            'train_labels'  : train_labels,
            'valid_dataset' : valid_dataset,
            'valid_labels'  : valid_labels,
            'test_dataset'  : test_dataset,
            'test_labels'   : test_labels
            }
    
    f = open(pickle_file, 'wb')
    pickle.dump (save, f, pickle.HIGHEST_PROTOCOL)
    f.close()

except Exception as e:
    print ('Unable to save data to %s' %pickle_file, ':' , e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
