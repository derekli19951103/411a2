import os
import urllib
from pylab import *
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import hashlib
import pickle

all_act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
act_nickname = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
act_labels = {'bracco': [1, 0, 0, 0, 0, 0], 'gilpin': [0, 1, 0, 0, 0, 0], 'harmon': [0, 0, 1, 0, 0, 0],
              'baldwin': [0, 0, 0, 1, 0, 0], 'hader': [0, 0, 0, 0, 1, 0], 'carell': [0, 0, 0, 0, 0, 1]}


def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray / 255.


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result


def get_images():
    testfile = urllib.URLopener()  # python 2

    if not os.path.exists("uncropped"):
        os.makedirs("uncropped")
    if not os.path.exists("cropped"):
        os.makedirs("cropped")
    if not os.path.exists("cropped_colored"):
        os.makedirs("cropped_colored")

    for a in all_act:
        name = a.split()[1].lower()
        i = 0
        for line in open("faces_subset.txt"):
            if a in line:
                filename = name + str(i) + '.' + line.split()[4].split('.')[-1]
                timeout(testfile.retrieve, (line.split()[4], "uncropped/" + filename), {}, 30)
                if not os.path.isfile("uncropped/" + filename):
                    continue
                try:
                    x1, y1, x2, y2 = line.split()[-2].split(',')
                    pic = imread('uncropped/' + filename)
                    image_file = open('uncropped/' + filename, 'rb').read()
                    if line.split()[-1] == hashlib.sha256(image_file).hexdigest():
                        pic = pic[int(y1):int(y2), int(x1):int(x2)]
                        try:
                            pic2 = imresize(pic, (224, 224))
                            imsave("cropped_colored/" + filename, pic2)
                            # pic1 = rgb2gray(pic)
                            # pic1 = imresize(pic1, (32, 32))
                            # imsave("cropped/" + filename, pic1)
                            print(filename)
                        except (ValueError, IndexError) as e:
                            pass
                except IOError:
                    pass
                i += 1
    print("done downloading")

    files = os.listdir("uncropped")
    names_set_u = {}
    for file in files:
        name = ''.join([i for i in file if not i.isdigit()]).split('.')[0]
        if name not in names_set_u:
            names_set_u[name] = [file]
        else:
            names_set_u[name].append(file)
    print("uncropped:")
    for name, files in names_set_u.items():
        print(name, len(files))
    files = os.listdir("cropped")
    names_set_c = {}
    for file in files:
        name = ''.join([i for i in file if not i.isdigit()]).split('.')[0]
        if name not in names_set_c:
            names_set_c[name] = [file]
        else:
            names_set_c[name].append(file)
    print("cropped:")
    for name, files in names_set_c.items():
        print(name, len(files))


def group(dir_name):
    files = os.listdir(dir_name)
    names_set = {}
    for file in files:
        name = ''.join([i for i in file if not i.isdigit()]).split('.')[0]
        if name not in names_set:
            names_set[name] = [file]
        else:
            names_set[name].append(file)
    training = []
    validating = []
    testing = []
    count = 0
    for name, files in names_set.items():
        for f in files:
            if count < 70:
                training.append(f)
                count += 1
            if 70 <= count < 80:
                validating.append(f)
                count += 1
            if 80 <= count < 90:
                testing.append(f)
                count += 1
        count = 0
    return names_set, training, validating, testing


def generate_x_y(setname):
    names_set, training, validating, testing = group('cropped')
    if setname == 'training':
        start = 0
        end = 60
    if setname == 'validating':
        start = 60
        end = 70
    if setname == 'testing':
        start = 70
        end = 90
    y = []
    x = ones((1, 1024))
    for name, files in names_set.items():
        if name in act_nickname:
            for i in files[start:end]:
                y.append(act_labels[name])
                pic = imread('cropped/' + i).flatten() / 255.
                x = vstack((x, pic))
    y = array(y).T
    x = np.delete(x, 0, 0)
    x = x.T
    return x, y


def generate_x_y_alexnet(setname):
    names_set, training, validating, testing = group('cropped_colored')
    if setname == 'training':
        start = 0
        end = 60
    if setname == 'validating':
        start = 60
        end = 70
    if setname == 'testing':
        start = 70
        end = 90
    y = []
    x = zeros(((end - start) * 6, 3, 224, 224))
    index = 0
    for name, files in names_set.items():
        if name in act_nickname:
            for i in files[start:end]:
                try:
                    pic = imread('cropped_colored/' + i)[:, :, :3]
                    pic = pic - np.mean(pic.flatten())
                    pic = pic / np.max(np.abs(pic.flatten()))
                    pic = np.rollaxis(pic, -1).astype(np.float32)
                    x[index] = pic
                    index += 1
                except IndexError:
                    pass
            for i in range(start, end):
                y.append(act_labels[name])

    y = array(y)
    if setname == 'testing':
        x = np.delete(x, 77, axis=0)
        y = np.delete(y, 77, axis=0)
    if setname == 'training':
        x = np.delete(x, 74, axis=0)
        y = np.delete(y, 74, axis=0)
    return x, y


if __name__ == "__main__":
    # get_images()
    x, y = generate_x_y_alexnet("training")
    print("End of Downloading")
