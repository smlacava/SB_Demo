import matplotlib.pyplot as plt
import tensorflow as tf
import math
from IPython.display import clear_output
import os
import numpy as np
import cv2
from skimage.transform import rotate, resize
import logging
from PIL import Image
import imageio
from MobileNet import MobileNet
import platform
from keras.models import load_model

for name in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.CRITICAL)


class SpoofBuster():
    def __init__(self, net=None, extractor='mindtct', live_label=1):
        if net is None:
            # net = MobileNet(n_classes=2)
            # net = tf.keras.models.load_model(os.path.dirname(os.path.realpath(__file__))+"\GreenBit")
            try:
              try:
                net = load_model(os.path.dirname(os.path.realpath(__file__)) + ".\GreenBit\my_model.h5")
              except:
                net = load_model(".\GreenBit\my_model.h5")
            except:
              net = None
        self.set_model(net)
        self._extractor = self.set_extractor(extractor)
        self.set_live_label(live_label)
        if extractor == 'mindtct':
            os.chmod(os.path.dirname(__file__) + '/mindtct', 0o755)

    def _load(self, filename, patch_size):
        np_image = Image.open(filename)
        np_image = np.array(np_image).astype('float32')
        np_image = resize(np_image, (patch_size[0], patch_size[1], 3))
        np_image = np.expand_dims(np_image, axis=0)
        return np_image

    def set_live_label(self, label=1):
        self._live_label = label

    def spoof_detection(self, fname, view=False, min_quality=0.0):
        """
    This method evaluates the spoofness probability of a single fingerprint,
    eventually showing which minutiae are considered as live and as spoof in
    detail.
    """
        if isinstance(fname, str):
            im = Image.open(fname).resize((500, 500))
            fname = [x for x in fname.split(".")][0]
        else:
            im = fname
            fname = "tmp"
        im.convert('L').save(fname + ".jpeg", "JPEG")
        self.extract_minutiae(fname + ".jpeg")
        [start_xs, end_xs, start_ys, end_ys, angle, qualities, types] = self.read_minutiae(fname + ".min")
        img = cv2.imread(fname + ".jpeg", cv2.IMREAD_GRAYSCALE)
        dim = np.shape(img)
        # img = bytearray(open(fname + '.brw', 'rb').read())
        img = np.reshape(img, dim)
        N = np.shape(start_xs)[0]
        scores = []
        tot_score = 0
        to_del = []
        count = 0
        patch_size = (224, 224)
        px = patch_size[0] * patch_size[1]
        for idx in range(N):
            if float(qualities[idx]) >= min_quality:
                try:
                    patch = self.extract_patch(img, start_xs[idx], start_ys[idx])
                    patch = self.resize(self.crop(self.align(patch, angle[idx])))
                    if self._check_values(patch, px):
                        print("Minutiae " + self._define_number(idx) + " of fingerprint " + fname + " not usable.")
                        to_del.append(idx)
                        continue
                    tmp = np.array(patch).astype('float32')
                    tmp = resize(tmp, (patch_size[0], patch_size[1], 3))
                    tmp = np.expand_dims(tmp, axis=0)
                    s = self._model.predict(tmp)
                    s = np.array(s).tolist()[0][0]
                    scores.append(s)
                    tot_score += s
                    count += 1
                except:
                    to_del.append(idx)
                clear_output()
            else:
                to_del.append(idx)
        if view is True:
            x = start_xs
            x = np.delete(x, to_del)
            y = start_ys
            y = np.delete(y, to_del)
            spoof_mask, live_mask = self._compute_masks(scores)
            plt.plot(x[spoof_mask], y[spoof_mask], 's', markeredgecolor='red',
                     markerfacecolor='None')
            plt.plot(x[live_mask], y[live_mask], 'o', markeredgecolor='green',
                     markerfacecolor='None')
            bin_img = bytearray(open(fname + '.brw', 'rb').read())
            bin_img = np.reshape(bin_img, (500, 500))
            plt.imshow(bin_img, alpha=0.5, cmap='gray')
            plt.show()
        return round(tot_score / count, 4)

    def _compute_masks(self, scores):
        if self._live_label == 0:
            spoof_mask = [s >= 0.5 for s in scores]
            live_mask = [s < 0.5 for s in scores]
        else:
            spoof_mask = [s <= 0.5 for s in scores]
            live_mask = [s > 0.5 for s in scores]
        return spoof_mask, live_mask

    def extract_minutiae(self, fname):
        """
    This method detects minutiae from a fingerprint.
    """
        if '.' in fname:
            aux_name = [x for x in fname.split('.')][0]
        else:
            aux_name = fname
            fname += ".jpeg"
        if 'Windows' in platform.system():
            m = os.path.dirname(__file__) + os.sep + 'nbis' + os.sep + 'bin' + os.sep + 'mindtct.exe'
            os.chmod(m, 0o755)
            os.system(m + ' -m1 -b ' + fname + ' ' + aux_name)
        else:
            m = os.path.dirname(__file__) + os.sep + 'mindtct'
            os.chmod(m, 0o755)
            os.system(m + ' -m1 -b ' + fname + ' ' + aux_name)

    def extract_patch(self, image, x, y, x_end=False, y_end=False,
                      patch_size=(136, 136)):
        """
    This method is used to extract the patches around each minutia of a
    fingerprint.
    """
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        x = self._minutiae_center(x, x_end)
        y = self._minutiae_center(y, y_end)
        dx = int(patch_size[0] / 2)
        dy = int(patch_size[1] / 2)
        image = np.mat(image)
        return image[x - dx:x + dx, y - dy:y + dy]

    def crop(self, patch, patch_size=(96, 96)):
        """
    This method crops the patch aruond a minutia.
    """
        sz = np.shape(patch)
        x = int(sz[0] / 2)
        y = int(sz[1] / 2)
        dx = int(patch_size[0] / 2)
        dy = int(patch_size[1] / 2)
        return patch[x - dx:x + dx, y - dy:y + dy]

    def align(self, patch, angle):
        """
    This method alignes a minutia based on its angle.
    """
        return rotate(patch, angle, resize=True, preserve_range=True)

    def resize(self, patch, patch_size=(224, 224)):
        """
    This method resizes a patch through bilinear interpolation.
    """
        return cv2.resize(patch, patch_size)

    def fit_model(self, x, y=None, ep=10, pat=3, val=0.1,
                  _current=False, val_data=None, class_weights=None, opt='adam'):
        """
    This method fits the CNN model for making it able it to detect if a minutia
    is live and spoof.
    """
        tf.config.run_functions_eagerly(True)
        if _current is False:
            mnet = MobileNet(n_classes=2)
            self._model = mnet
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                        patience=pat,
                                                        restore_best_weights=True)
            self._model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        if val_data is None:
            if y is None:
                self._model.fit(x, validation_split=val, batch_size=100, epochs=ep, verbose=1,
                                class_weight=class_weights,
                                callbacks=[callback])
            else:
                self._model.fit(x, y, batch_size=100, epochs=ep, validation_split=val, verbose=1,
                                class_weight=class_weights,
                                callbacks=[callback])
        else:
            if y is None:
                self._model.fit(x, validation_data=val_data, batch_size=100, epochs=ep, verbose=1,
                                class_weight=class_weights,
                                callbacks=[callback])
            else:
                self._model.fit(x, y, batch_size=100, epochs=ep, validation_data=val_data, verbose=1,
                                callbacks=[callback])

    def save_patches(self, fname, outDir, min_quality=0.0):
        """
    This method saves all the patches already extracted around minutiae as
    single images.
    """
        aux_fname = [x for x in fname.split(".")][0]
        try:
            [start_xs, end_xs, start_ys, end_ys, angle, qualities, types] = self.read_minutiae(aux_fname + ".min")
        except:
            print(aux_fname + " is not usable")
            return
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        dim = np.shape(img)
        # img = bytearray(open(aux_fname + '.jpeg', 'rb').read())
        img = np.reshape(img, dim)
        N = np.shape(start_xs)[0]
        sub = [x for x in aux_fname.split("_")][0]
        patch_size = (224, 224)
        dp = (int(patch_size[0] / 2), int(patch_size[1] / 2))
        px = patch_size[0] * patch_size[1]
        for idx in range(N):
            try:
                if float(qualities[idx]) >= min_quality:
                    if self._check_edges(start_xs[idx], end_xs[idx], start_ys[idx], end_ys[idx], dim, dp) is True:
                        patch = self.extract_patch(img, np.floor((start_xs[idx] + end_xs[idx]) / 2),
                                                   np.floor((start_ys[idx] + end_ys[idx]) / 2))
                        patch = self.align(patch, angle[idx])
                        patch = self.crop(patch)
                        patch = self.resize(patch, patch_size)
                        imageio.imwrite(outDir + [x for x in aux_fname.split(os.sep)][-1] + "_" + str(idx) + ".jpeg",
                                        patch)
                        patch = cv2.imread(outDir + [x for x in aux_fname.split(os.sep)][-1] + "_" + str(idx) + ".jpeg",
                                           cv2.IMREAD_GRAYSCALE)
                        if self._check_values(patch, px):
                            print("Minutiae " + self._define_number(
                                idx) + " of fingerprint " + aux_fname + " not usable.")
                            os.remove(outDir + [x for x in aux_fname.split(os.sep)][-1] + "_" + str(idx) + ".jpeg")
                    else:
                        print("Minutiae " + self._define_number(
                            idx) + " of fingerprint " + aux_fname + " is too close to edges.")
            except:
                print("Minutiae " + self._define_number(idx) + " of fingerprint " + aux_fname + " not usable.")

    def _check_values(self, img, px):
        val = int(np.sum(np.sum(img)) / px)
        return val <= 1 or val >= 254

    def _define_number(self, idx):
        if (idx + 1) % 10 == 1:
            return str(idx + 1) + "st"
        elif (idx + 1) % 10 == 2:
            return str(idx + 1) + "nd"
        elif (idx + 1) % 10 == 3:
            return str(idx + 1) + "rd"
        else:
            return str(idx + 1) + "th"

    def get_model(self):
        """
    This method returns the current model.
    """
        return self._model

    def set_model(self, model):
        """
    This method allows to set the already fitted network model which has to be 
    used by the system.
    """
        self._model = model

    def set_extractor(self, extractor):
        self._extractor = extractor

    def read_minutiae(self, filename):
        """
    This method reads the already detected minutiae within a fingerprint, 
    returning their starting coordinates, their ending coordinates, their angles
    and their types.
    """
        with open(filename) as f:
            minutiae_temp = [line.strip().split(':') for line in f][4:]
        coords = np.array([ln[1].split(',') for ln in minutiae_temp], dtype=int)
        xs = coords[:, 0]
        ys = coords[:, 1]
        angles = np.array([ln[2] for ln in minutiae_temp], dtype=int)
        qualities = np.array([ln[3].strip() for ln in minutiae_temp])
        types = np.array([ln[4].strip() for ln in minutiae_temp])

        num_minutiae = len(xs)
        line_len = 20
        start_xs = np.array(xs)
        start_ys = np.array(ys)
        angles_rad = angles * 11.25 / 180.0 * math.pi
        end_xs = np.array([start_xs[ind] + line_len * math.sin(angles_rad[ind]) for ind in range(0, num_minutiae)])
        end_ys = np.array([start_ys[ind] - line_len * math.cos(angles_rad[ind]) for ind in range(0, num_minutiae)])

        return start_xs, end_xs, start_ys, end_ys, angles, qualities, types

    def _minutiae_center(self, val, val_end):
        """
    This method searches for the central coordinate of a minutia (FOR INTERNAL
    USE ONLY)
    """
        if not (val_end is False):
            val = (val + val_end) / 2
        return int(val)

    def _check_edges(self, x0, x1, y0, y1, dim, dp):
        """
    This method checks if the patch is within image edges (FOR INTERNAL USE ONLY)
    """
        if x0 - dp[0] < 0 or y0 - dp[1] < 0 or x1 + dp[0] > dim[0] or y1 + dp[1] > dim[1]:
            return False
        else:
            return True