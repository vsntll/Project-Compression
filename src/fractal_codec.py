import pywt
import numpy as np
import cv2
import os

def wavelet_compress(img, wavelet='haar', level=3, quant_step=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coeffs = pywt.wavedec2(gray, wavelet=wavelet, level=level)

    coeffs_quant = []
    for arr in coeffs:
        if isinstance(arr, tuple):
            quant_arr = tuple(np.round(c / quant_step).astype(np.int16) for c in arr)
            coeffs_quant.append(quant_arr)
        else:
            quant_arr = np.round(arr / quant_step).astype(np.int16)
            coeffs_quant.append(quant_arr)
    return coeffs_quant

def wavelet_save(coeffs_quant, save_path):
    to_save = {}
    to_save['cA'] = coeffs_quant[0]
    for i, detail_coeffs in enumerate(coeffs_quant[1:], start=1):
        to_save[f'cH_{i}'], to_save[f'cV_{i}'], to_save[f'cD_{i}'] = detail_coeffs
    np.savez_compressed(save_path, **to_save)

def wavelet_load(load_path):
    data = np.load(load_path)
    cA = data['cA']
    coeffs = [cA]
    i = 1
    while f'cH_{i}' in data:
        cH = data[f'cH_{i}']
        cV = data[f'cV_{i}']
        cD = data[f'cD_{i}']
        coeffs.append((cH, cV, cD))
        i += 1
    return coeffs

def wavelet_decompress(coeffs_quant, wavelet='haar', quant_step=10):
    coeffs_dequant = []
    for arr in coeffs_quant:
        if isinstance(arr, tuple):
            dequant_arr = tuple((c.astype(np.float32) * quant_step) for c in arr)
            coeffs_dequant.append(dequant_arr)
        else:
            dequant_arr = (arr.astype(np.float32) * quant_step)
            coeffs_dequant.append(dequant_arr)
    img_rec = pywt.waverec2(coeffs_dequant, wavelet=wavelet)
    img_rec = np.clip(img_rec, 0, 255).astype(np.uint8)
    return img_rec

# Usage example
if __name__ == "__main__":
    img = cv2.imread('path/to/image.png')
    coeffs_comp = wavelet_compress(img)
    wavelet_save(coeffs_comp, 'compressed_img.npz')
    loaded_coeffs = wavelet_load('compressed_img.npz')
    reconstructed_img = wavelet_decompress(loaded_coeffs)
    cv2.imwrite('reconstructed_wavelet.png', reconstructed_img)
