import cv2 as cv
import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class SeamCarver:
    def __init__(self, input_img, specifications) -> None:
        self.img_h, self.img_w = len(input_img), len(input_img[0])
        self.specifications = specifications
        self.reqd_h = specifications["height"]
        self.reqd_w = specifications["width"]
        self.input_img = input_img

    def validate(self):
        print(f"Input Dimensions: {self.img_w} x {self.img_h}")
        print(f"Required Dimensions: {self.reqd_w} x {self.reqd_h}")

        if self.reqd_h > self.img_h or self.reqd_w > self.img_w:
            return False
        return True

    def vertical_seam_carving(self):
        self.energy_map = np.array(
            [[0 for __ in range(self.img_w)] for _ in range(self.img_h)])

        vert = np.abs(self.input_img[2:, 1:-1] - self.input_img[:-2, 1:-1])
        horiz = np.abs(self.input_img[1:-1, 2:] - self.input_img[1:-1, :-2])

        # sum energy over the 3 color channels
        total = vert.sum(axis=2) + horiz.sum(axis=2)
        self.energy_map[1:-1, 1:-1] += total.astype(np.int64)

        # --- normalization ---
        em = self.energy_map.astype(np.float64)
        emin, emax = em.min(), em.max()

        if emax > emin:
            norm = (em - emin) / (emax - emin)
        else:
            norm = np.zeros_like(em)

        plt.imshow(norm, cmap='gray')
        plt.colorbar()
        plt.title("Normalized Energy Map")
        plt.show()

    def run_horizontal_seam_carving(self):
        pass


def main():
    input_img = cv.imread("data/dog_image.jpg")

    with open("data/specifications.json") as f:
        specifications = json.load(f)

    obj = SeamCarver(input_img, specifications)

    if not obj.validate():
        print("Required dimensions are larger than image dimensions.")
        return

    obj.run_horizontal_seam_carving()
    obj.vertical_seam_carving()


if __name__ == '__main__':
    main()
