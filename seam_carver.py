import matplotlib.pyplot as plt
import cv2 as cv
import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")


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
        img = cv.cvtColor(self.input_img, cv.COLOR_BGR2GRAY).astype(np.float64)
        num_seams = self.img_w - self.reqd_w
        self.output_img = self.input_img.copy()

        for i in range(num_seams):
            H = self.img_h
            W = self.img_w - i

            print("Pass", i + 1)
            gx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)  # d/dx
            gy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)  # d/dy

            energy_map = (np.abs(gx) + np.abs(gy)).astype(np.float64)
            dp = energy_map.copy()

            # Form DP table
            for row in range(1, H):
                for col in range(W):
                    if col == 0:
                        dp[row, col] += min(dp[row-1, col], dp[row-1, col+1])
                    elif col == W - 1:
                        dp[row, col] += min(dp[row-1, col-1], dp[row-1, col])
                    else:
                        dp[row, col] += min(dp[row-1, col-1],
                                            dp[row-1, col], dp[row-1, col+1])

            seam = np.zeros(img.shape[0], dtype=np.int32)
            seam[-1] = np.argmin(dp[-1])

            for row in range(H - 2, -1, -1):
                prev = seam[row + 1]  # start from the bottom seam position
                if prev == 0:
                    choices = [dp[row, prev], dp[row, prev + 1]]
                    seam[row] = prev + np.argmin(choices)
                elif prev == W - 1:
                    choices = [dp[row, prev - 1], dp[row, prev]]
                    seam[row] = prev - 1 + np.argmin(choices)
                else:
                    choices = [dp[row, prev - 1],
                               dp[row, prev], dp[row, prev + 1]]
                    seam[row] = prev - 1 + np.argmin(choices)

            img = np.array([np.delete(img[row], seam[row])
                           for row in range(H)])
            self.output_img = np.array(
                [np.delete(self.output_img[row], seam[row], axis=0) for row in range(H)])

        plt.imshow(cv.cvtColor(self.output_img, cv.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig("output.png", bbox_inches='tight', pad_inches=0)
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

    obj.vertical_seam_carving()
    obj.run_horizontal_seam_carving()

if __name__ == '__main__':
    main()
